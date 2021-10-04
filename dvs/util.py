import os
import torch
import cv2
from itertools import chain
from warp import load_video, save_video
import numpy as np
import matplotlib.pyplot as plt
from gyro import get_rotations
import shutil

def save_train_info(name, checkpoints_dir, cf, model, count, optimizer = None):
    path = None
    if name == "last":
        path = os.path.join(checkpoints_dir, cf['data']['exp']+'_last.checkpoint')
    elif name == "best":
        path = os.path.join(checkpoints_dir, cf['data']['exp']+'_best.checkpoint')
    else:
        path = os.path.join(checkpoints_dir, cf['data']['exp']+'_epoch%d.checkpoint'%count)
    torch.save(model.save_checkpoint(epoch = count, optimizer=optimizer), path)

def make_dir(checkpoints_dir ,cf):
    inference_path = "./test"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(cf["data"]["log"]):
        os.makedirs(cf["data"]["log"])
    if not os.path.exists(inference_path):
        os.makedirs(inference_path)
        
    inference_path = os.path.join(inference_path, cf['data']['exp'])
    if not os.path.exists(inference_path):
        os.makedirs(inference_path)
    checkpoints_dir = os.path.join(checkpoints_dir, cf['data']['exp'])
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    return checkpoints_dir

def get_optimizer(optimizer, model, init_lr, cf):
    if optimizer == "adam":
        optimizer = torch.optim.Adam(chain(model.net.parameters(), model.unet.parameters()), lr=init_lr, weight_decay=cf["train"]["weight_decay"])
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(chain(model.net.parameters(), model.unet.parameters()), lr=init_lr, momentum=cf["train"]["momentum"])
    return optimizer

def crop_video(in_path, out_path, crop_ratio):
    frame_array, fps, size = load_video(in_path)
    hs = int((1-crop_ratio)*1080) + 1
    he = int(crop_ratio*1080) - 1
    ws = int((1-crop_ratio)*1920) + 1
    we = int(crop_ratio*1920) - 1
    for i in range(len(frame_array)):
        frame_array[i] = cv2.resize(frame_array[i][hs:he,ws:we,:], size, interpolation = cv2.INTER_LINEAR)
    save_video(out_path, frame_array, fps, size= size)

def norm_flow(flow, h, w):
    if flow.shape[2] == 2:
        flow[:,:,0] /= h
        flow[:,:,1] /= w
    else:
        flow[:,:,:,0] /= h
        flow[:,:,:,1] /= w
    return flow

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        if self.cnt > 0:
            self.avg = self.sum / self.cnt