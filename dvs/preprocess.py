from flownet2 import flow_utils
import os
from scipy import ndimage, misc
import numpy as np
import shutil
from warp import save_video, load_video
import cv2


def process_video(path):
    process_flo(os.path.join(path, "flo"))
    process_flo(os.path.join(path, "flo_back"))

def copy_video(src, dst):
    os.makedirs(dst)
    files = sorted(os.listdir(src))
    for f in files:
        f_src = os.path.join(src, f)
        if os.path.isfile(f_src):
            f_dst = os.path.join(dst, f)
            shutil.copyfile(f_src, f_dst)


def process_flo(path, ratio = 0.25):
    frame = sorted(os.listdir(path))
    for i in range(len(frame)):
        f_path = os.path.join(path, frame[i])
        print(f_path)
        f = flow_utils.readFlow(f_path).astype(np.float32) 
        f = resize_flow(f, ratio = ratio)
        flow_utils.writeFlow( f_path,  f)

def resize_flow(flow, ratio):
    f0 = np.expand_dims(ndimage.zoom(flow[:,:,0], ratio), axis = 2)
    f1 = np.expand_dims(ndimage.zoom(flow[:,:,1], ratio), axis = 2)
    return np.concatenate((f0,f1),axis=2)
    
def crop_frame(frame, h = 1080, w = 1920, ratio = 0.1):
    h_low = int(h*ratio)
    h_upp = int(h*(1-ratio))
    w_low = int(w*ratio)
    w_upp = int(w*(1-ratio))
    
    frame = frame[h_low:h_upp+1,w_low:w_upp+1,:]
    frame = cv2.resize(frame, (1920, 1080), interpolation = cv2.INTER_LINEAR)
    return frame

def crop_video(video_path, save_path):
    frames, fps, size = load_video(video_path)
    w = size[0]
    h = size[1]
    # for i in range(len(frames)):
    #     frames[i] = crop_frame(frames[i], h = h, w = w, ratio = 0.10)
    save_video(save_path, frames, fps = 30, size = (1920, 1080))

# path1 = "/mnt/disks/dataset/Google/train"
# path2 = "/mnt/disks/dataset/Google_raw"
# videos = sorted(os.listdir(path1))
# for i in range(len(videos)):
#     v_path1 = os.path.join(path1, videos[i])
#     v_path2 = os.path.join(path2, videos[i])
#     copy_video(v_path1, v_path2)
path1 = "/home/zhmeishi_google_com/dvs/test/opt_base3_continue"
path2 = "/home/zhmeishi_google_com/presentation/indoor_opt"

# path1 = "/home/zhmeishi_google_com/CVPR2020CODE/result_google"
# path1 = "/home/zhmeishi_google_com/presentation/opt_base3_continue_75_20_5/s2_outdoor_runing_forward_VID_20200304_144434_stab.mp4"
# path2 = "/home/zhmeishi_google_com/presentation/opt_base3_continue_75_20_5/s2_outdoor_runing_forward_VID_20200304_144434_stab_fps60.mp4"
# crop_video(path1, path2)
videos = sorted(os.listdir(path1))
for i in range(len(videos)):
    if videos[i][-3:] == "mp4":
        print(videos[i])
        v_path1 = os.path.join(path1, videos[i])
        v_path2 = os.path.join(path2, videos[i])
        crop_video(v_path1, v_path2)
        
    