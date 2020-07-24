from utils import flow_utils
import os
from scipy import ndimage, misc
import numpy as np


def process_video(path):
    process_flo(os.path.join(path, "flo"))
    process_flo(os.path.join(path, "flo_back"))

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

path = "/mnt/disks/dataset/Google/transfer"
videos = sorted(os.listdir(path))
for i in range(len(videos)):
    v_path = os.path.join(path, videos[i])
    process_video(v_path)