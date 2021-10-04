import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

import time
import yaml
import argparse
import numpy as np
from printer import Printer
from dataset import get_data_loader, get_inference_data_loader
from model import Model
import datetime
import copy
from util import make_dir, get_optimizer, norm_flow
from gyro import (
    get_grid, 
    get_rotations, 
    visual_rotation, 
    GetGyroAtTimeStamp, 
    torch_ConvertQuaternionToAxisAngle, 
    torch_ConvertAxisAngleToQuaternion,
    torch_QuaternionProduct,
    get_static
    )
from warp import warp_video

def run(loader, cf, USE_CUDA=True):
    number_virtual, number_real = cf['data']["number_virtual"], cf['data']["number_real"]
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real_inputs, times, flo, flo_back, real_projections, real_postion, ois, real_queue_idx = data
        print("Fininsh Load data")

        real_inputs = real_inputs.type(torch.float) #[b,60,84=21*4]
        real_projections = real_projections.type(torch.float) 
    
        batch_size, step, dim = real_inputs.size()
        times = times.numpy()
        real_queue_idx = real_queue_idx.numpy()
        virtual_queue = [None] * batch_size

        for j in range(step):
            virtual_inputs, vt_1 = loader.dataset.get_virtual_data(
                virtual_queue, real_queue_idx, times[:, j], times[:, j+1], times[:, 0], batch_size, number_virtual, real_postion[:,j]) 
            real_inputs_step = real_inputs[:,j,:]
            if USE_CUDA:
                real_inputs_step = real_inputs_step.cuda()
                virtual_inputs = virtual_inputs.cuda()
                real_postion_anchor = real_postion[:,j].cuda()

            out = real_inputs_step[:,40:44]
            
            virtual_position = virtual_inputs[:, -4:]
            pos = torch_QuaternionProduct(virtual_position, real_postion_anchor)

            out = torch_QuaternionProduct(out, pos)

            if USE_CUDA:
                out = out.cpu().detach().numpy() 

            virtual_queue = loader.dataset.update_virtual_queue(batch_size, virtual_queue, out, times[:,j+1])
    return np.squeeze(virtual_queue, axis=0)

def inference(cf, data_path, USE_CUDA):
    print("-----------Load Dataset----------")
    test_loader = get_inference_data_loader(cf, data_path)
    data = test_loader.dataset.data[0]
    test_loader.dataset.no_flo = True
    test_loader.dataset.static_options = get_static(ratio = 0)

    start_time = time.time()
    virtual_queue = run(test_loader, cf, USE_CUDA=USE_CUDA)

    virtual_data = np.zeros((1,5))
    virtual_data[:,1:] = virtual_queue[0, 1:]
    virtual_data[:,0] = data.frame[0,0]
    virtual_queue = np.concatenate((virtual_data, virtual_queue), axis = 0)
    
    files = os.listdir(data_path)
    for f in files:
        if f[-3:] == "mp4" and "no_ois" not in f and "gimbal" not in f.lower():
            video_name = f[:-4]
            print(video_name)
    virtual_path = os.path.join("./test", cf['data']['exp'], video_name+'.txt')

    print("------Start Visual Result--------")
    rotations_real, lens_offsets_real = get_rotations(data.frame[:data.length], data.gyro, data.ois, data.length)
    fig_path = os.path.join(data_path, video_name+"_real.jpg")
    visual_rotation(rotations_real, lens_offsets_real, None, None, None, None, fig_path)
    
    return

def main(args = None):
    config_file = args.config
    dir_path = args.dir_path
    cf = yaml.load(open(config_file, 'r'))
    
    USE_CUDA = cf['data']["use_cuda"]

    checkpoints_dir = cf['data']['checkpoints_dir']
    checkpoints_dir = make_dir(checkpoints_dir, cf)

    data_name = sorted(os.listdir(dir_path))
    for i in range(len(data_name)):
        print("Running: " + str(i+1) + "/" + str(len(data_name)))
        inference(cf, os.path.join(dir_path, data_name[i]), USE_CUDA)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--config", default="./conf/stabilzation.yaml", help="Config file.")
    parser.add_argument("--dir_path", default="./video")
    args = parser.parse_args()
    main(args = args)