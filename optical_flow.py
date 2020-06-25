import cv2
import numpy as np
from bob.ip.optflow.liu.sor import flow
import time
import bob.io.image
import bob.io.video
import bob.io.base

p3 = "./data/testdata/inputs/s2_outdoor_runing_forward_VID_20200304_144434/s2_outdoor_runing_forward_VID_20200304_144434.mp4"

def optical_flow_cv2_show(path):
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()

def optical_flow_cv2(path1, path2):
    frame1 = cv2.imread(path1)
    frame2 = cv2.imread(path2)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    t1 = time.time()
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    t2 = time.time()
    print(t2-t1)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    cv2.imwrite('flow_cv2.jpg',rgb)

def optical_flow_liu(path1, path2):
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30

    i1 = bob.io.base.load(p1).astype('float64')/255.
    i2 = bob.io.base.load(p2).astype('float64')/255.

    t1 = time.time()
    (u, v, warped) = flow(i1, i2, alpha=alpha, ratio=ratio, min_width = minWidth, n_outer_fp_iterations= \
        nOuterFPIterations, n_inner_fp_iterations= nInnerFPIterations, n_sor_iterations=nSORIterations)
    t2 = time.time()
    print(t2-t1)

    f = np.ones(i1.shape)*0.8
    u = u-np.min(u)
    v = v-np.min(v)
    f[2] = u/np.max(u)*2
    f[1] = v/np.max(v)*2
    f = np.transpose(f, (1,2,0))

    f = np.clip(f*255, 0, 255).astype("uint8")
    cv2.imwrite("flow_liu.jpg", f) 


if __name__ == "__main__":
    p3 = "./data/testdata/inputs/s2_outdoor_runing_forward_VID_20200304_144434/s2_outdoor_runing_forward_VID_20200304_144434.mp4"
    p1 = "OpticalFlow/car1.jpg"
    p2 = "OpticalFlow/car2.jpg"
    optical_flow_liu(p1,p2)
    # optical_flow_cv2(p1,p2)