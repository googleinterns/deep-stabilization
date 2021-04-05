import os
import sys
import numpy as np
import cv2
import math
import pdb
import matplotlib.pyplot as plt
from printer import Printer
from warp import video2frame_one_seq
import datetime
import torch
import copy
import csv
import copyreg
import shutil
import matplotlib.pyplot as plt
from util import crop_video

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

h_size = 480
w_size = 640

def crop_metric(M):
    points = np.array([[0,0,1],[0,h_size,1], [w_size,0,1], [w_size,h_size,1]]).T
    result = np.matmul(M,points).T
    result = result[:,:2]/result[:,2:]
    w_out = 1 - max(result[0,0], result[1,0], w_size - result[2,0], w_size - result[3,0], 0)/w_size
    h_out = 1 - max(result[0,1], result[2,1], h_size - result[1,1], h_size - result[3,1], 0)/h_size
    return w_out, h_out

# https://stackoverflow.com/questions/34389125/how-to-get-the-scale-factor-of-getperspectivetransform-in-opencv
def get_scale(M):
    h1 = M[0, 0]
    h2 = M[0, 1]
    h3 = M[0, 2]
    h4 = M[1, 0]
    h5 = M[1, 1]
    h6 = M[1, 2]
    h7 = M[2, 0]
    h8 = M[2, 1]
    QR = np.array([[h1-(h7*h3), h2-(h8*h3)], [h4-(h7*h6), h5-(h8*h6)]])
    Q, R = np.linalg.qr(QR)
    return abs(R[0,0]), abs(R[1,1])

# https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image
def get_rescale_matrix(M, sx, sy):
    S = np.eye(3, dtype = float)
    S[0,0] = sx
    S[1,1] = sy

    S1 = np.eye(3, dtype = float)
    S1[0,0] = 1/sx
    S1[1,1] = 1/sy
    return np.matmul(M, S1)

# Part of code reference from https://github.com/jinsc37/DIFRINT/blob/master/metrics.py
def metrics(in_src, out_src, package, crop_scale = False, re_compute = False):
    load_dic = None
    if re_compute and os.path.exists(package):
        print("Start load")
        load_dic = torch.load(package)
        print("Finish load")
    dic = {
        'M': None,
        'CR_seq': [],
        'DV_seq': [],
        'SS_t': None,
        'SS_r': None,
        'w_crop':[],
        'h_crop':[],
        'distortion': [],
        'count': 0,
        'in_sift': {},
        'out_sift': {},
        'fft_t': {},
        'fft_r': {}
        }

    if load_dic is not None:
        dic["in_sift"] = load_dic["in_sift"]
        dic["out_sift"] = load_dic["out_sift"]

    frameList_in = sorted(os.listdir(in_src))
    frameList = sorted(os.listdir(out_src))
    frameList = frameList[:min(len(frameList_in),len(frameList))]

    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    # Apply the homography transformation if we have enough good matches 
    MIN_MATCH_COUNT = 10 #10

    ratio = 0.7 #0.7
    thresh = 5.0 #5.0

    Pt = np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    P_seq = []
    count = 1
    for index, f in enumerate(frameList, 0):
        if f.endswith('.png'):
            # Load the images in gray scale
            img1 = cv2.imread(os.path.join(in_src, f), 0)  
            img1 = cv2.resize(img1, (w_size,h_size), interpolation = cv2.INTER_LINEAR)

            img1o = cv2.imread(os.path.join(out_src, f), 0)
            img1o = cv2.resize(img1o, (w_size,h_size), interpolation = cv2.INTER_LINEAR)
            sift = cv2.SIFT_create()   
            
            if f in dic["in_sift"]:
                keyPoints1, descriptors1 = dic["in_sift"][f]
            else:
                # Detect the SIFT key points and compute the descriptors for the two images
                keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
                dic["in_sift"][f] = (keyPoints1, descriptors1)

            if f in dic["out_sift"]:
                keyPoints1o, descriptors1o = dic["out_sift"][f]
            else:
                keyPoints1o, descriptors1o = sift.detectAndCompute(img1o, None)
                dic["out_sift"][f] = (keyPoints1o, descriptors1o)

            # Match the descriptors
            matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

            # Select the good matches using the ratio test
            goodMatches = []

            for m, n in matches:
                if m.distance < ratio * n.distance:
                    goodMatches.append(m)

            if len(goodMatches) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                destinationPoints = np.float32([ keyPoints1o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
                im_dst = cv2.warpPerspective(img1, M, (w_size,h_size))  

                cm = []
                for i in range(6):
                    for j in range(6):
                        hs = int(h_size * (0.2 + 0.1 * i))
                        he = int(h_size * (0.3 + 0.1 * i))
                        ws = int(w_size * (0.2 + 0.1 * j))
                        we = int(w_size * (0.3 + 0.1 * j))
                        cm.append(np.corrcoef(img1o[hs:he, ws:we].flat, im_dst[hs:he, ws:we].flat))
                dic["distortion"].append(cm)

                if crop_scale:
                    sx, sy = get_scale(M)
                    M_scale = get_rescale_matrix(M, sx, sy)
                    w_crop, h_crop = crop_metric(M_scale)
                else:
                    w_crop, h_crop = crop_metric(M)
                dic["w_crop"].append(w_crop)
                dic["h_crop"].append(h_crop)

            # Obtain Scale, Translation, Rotation, Distortion value
            sx = M[0, 0]
            sy = M[1, 1]
            scaleRecovered = math.sqrt(np.abs(sx*sy))

            w, _ = np.linalg.eig(M[0:2,0:2])
            w = np.sort(w)[::-1]
            DV = w[1]/w[0]
            #pdb.set_trace()

            dic["CR_seq"].append(1.0/scaleRecovered)
            dic["DV_seq"].append(DV)  

            # For Stability score calculation
            if count < len(frameList):
                f_path = f[:-9] + '%05d.png' % (int(f[-9:-4])+1)
                if f_path in dic["out_sift"]:
                    keyPoints2o, descriptors2o = dic["out_sift"][f_path]
                else:
                    img2o = cv2.imread(os.path.join(out_src, f_path), 0)
                    img2o = cv2.resize(img2o, (w_size,h_size), interpolation = cv2.INTER_LINEAR)
                    keyPoints2o, descriptors2o = sift.detectAndCompute(img2o, None)
                    dic["out_sift"][f_path] = (keyPoints2o, descriptors2o)
                
                matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)
                goodMatches = []

                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        goodMatches.append(m)

                if len(goodMatches) > MIN_MATCH_COUNT:
                    # Get the good key points positions
                    sourcePoints = np.float32([ keyPoints1o[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                    destinationPoints = np.float32([ keyPoints2o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
                    
                    # Obtain the homography matrix
                    M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)

                P_seq.append(np.matmul(Pt, M))
                Pt = np.matmul(Pt, M)
            if count % 10 ==0:
                sys.stdout.write('\rFrame: ' + str(count) + '/' + str(len(frameList)))
                sys.stdout.flush()
            dic["count"] = count
            count += 1

    # Make 1D temporal signals
    P_seq_t = np.asarray([1])
    P_seq_r = np.asarray([1])

    #pdb.set_trace()
    for Mp in P_seq:
        sx = Mp[0, 0]
        sy = Mp[1, 1]
        c = Mp[0, 2]
        f = Mp[1, 2]

        transRecovered = math.sqrt(c*c + f*f)
        thetaRecovered = math.atan2(sx, sy) * 180 / math.pi

        P_seq_t = np.concatenate((P_seq_t, [transRecovered]), axis=0)
        P_seq_r = np.concatenate((P_seq_r, [thetaRecovered]), axis=0)

    P_seq_t = np.delete(P_seq_t, 0)
    P_seq_r = np.delete(P_seq_r, 0)

    # FFT
    fft_t = np.fft.fft(P_seq_t)
    fft_r = np.fft.fft(P_seq_r)
    fft_t = abs(fft_t)**2
    fft_r = abs(fft_r)**2

    fft_t = np.delete(fft_t, 0)
    fft_r = np.delete(fft_r, 0)
    fft_t = fft_t[:int(len(fft_t)/2)]
    fft_r = fft_r[:int(len(fft_r)/2)]

    dic["fft_t"] = fft_t
    dic["fft_r"] = fft_r

    SS_t = np.sum(fft_t[:5])/np.sum(fft_t)  
    SS_r = np.sum(fft_r[:5])/np.sum(fft_r)

    dic["CR_seq"] = np.array(dic["CR_seq"])
    dic["DV_seq"] = np.array(dic["DV_seq"])
    dic["w_crop"] = np.array(dic["w_crop"])
    dic["h_crop"] = np.array(dic["h_crop"])
    dic["distortion"] = np.array(dic["distortion"])
    dic["SS_t"] = SS_t
    dic["SS_r"] = SS_r
    
    if not (re_compute and os.path.exists(package)):
        torch.save(dic, package)

    DV_seq = np.absolute(dic["DV_seq"])
    DV_seq = DV_seq[np.where((DV_seq >= 0.5) & (DV_seq <= 1))]
    Distortion = str.format('{0:.4f}', np.nanmin(DV_seq))
    Distortion_avg = str.format('{0:.4f}', np.nanmean(DV_seq))

    Trans = str.format('{0:.4f}', dic["SS_t"])
    Rot = str.format('{0:.4f}', dic["SS_r"])

    w_crop = crop_rm_outlier(dic["w_crop"])
    h_crop = crop_rm_outlier(dic["h_crop"])

    FOV = str.format( '{0:.4f}', min(np.nanmin(w_crop), np.nanmin(h_crop)) )
    FOV_avg = str.format( '{0:.4f}', (np.nanmean(w_crop)+np.nanmean(h_crop)) / 2 )

    Correlation_avg = str.format( '{0:.4f}', np.nanmean(dic["distortion"][10:]) )
    Correlation_min = str.format( '{0:.4f}', np.nanmin(dic["distortion"][10:]) )

    # Print results
    print('\n***Distortion value (Avg, Min):')
    print(Distortion_avg +' | '+  Distortion)
    print('***Stability Score (Avg, Trans, Rot):')
    print(str.format('{0:.4f}',  (dic["SS_t"]+dic["SS_r"])/2) +' | '+ Trans +' | '+ Rot )
    print("=================")
    print('***FOV ratio (Avg, Min):')
    print( FOV_avg +' | '+ FOV )
    print('***Correlation value (Avg, Min):')
    print( Correlation_avg +' | '+ Correlation_min , "\n")  

    dic['in_sift'] = 0
    dic['out_sift'] = 0
    torch.save(dic, package[:-3]+"_light.pt") 
    return float(FOV)

def crop_rm_outlier(crop):
    crop = np.array(crop)
    crop = crop[crop >= 0.5]
    return sorted(crop)[5:]

if __name__ == '__main__':
    metric_path = os.path.join("./test/stabilzation/metric")
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    in_video = "./video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820.mp4"
    in_folder = os.path.join(metric_path, "in_frame")
    if not os.path.exists(in_folder):
        os.makedirs(in_folder)
        print("Convert video to frames")
        video2frame_one_seq(in_video, in_folder)
        
    out_video = "./test/stabilzation/s_114_outdoor_running_trail_daytime_stab.mp4"
    out_folder = os.path.join(metric_path, "out_frame")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print("Convert video to frames")
        video2frame_one_seq(out_video, out_folder)
    
    package = os.path.join(metric_path, "stabilzation.pt")
    FOV = metrics(in_folder, out_folder, package)

    crop_path = out_video[:-4] + "_crop.mp4"
    crop_video(out_video, crop_path, FOV)
