import os
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
from dataset import get_data_loader, get_virtual_data, get_inference_data_loader
from model import Model
import datetime
import copy
from util import make_dir, get_optimizer, AverageMeter
from gyro import get_grid, get_rotations, visual_rotation
from warp import warp_video

def run(model, loader, cf, USE_CUDA=True):
    number_virtual, number_real = cf['data']["number_virtual"], cf['data']["number_real"]
    sample_freq = cf['data']["sample_freq"]
    avg_loss = AverageMeter()
    model.net.eval()
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real_inputs, times = data
        real_inputs = real_inputs.type(torch.float) #[b,60,84=21*4]
        batch_size, step, dim = real_inputs.size()
        virtual_queue = None
        for j in range(step):
            virtual_inputs = get_virtual_data(virtual_queue, times[:, j], batch_size, number_virtual, sample_freq) # [b,40=4*10]

            inputs = torch.cat((real_inputs[:,j,:],virtual_inputs), dim = 1)
            inputs = Variable(inputs)
            if USE_CUDA:
                real_inputs_step = real_inputs[:,j,4*number_real:4*number_real+4].cuda()
                virtual_inputs = virtual_inputs.cuda()
                inputs = inputs.cuda()

            with torch.no_grad():
                out = model.net(inputs)
        
            loss = model.loss(out, virtual_inputs, real_inputs_step)

            avg_loss.update(loss.item(), batch_size) 
            
            if USE_CUDA:
                out = out.cpu()
            out = out.detach().numpy()

            virtual_data = np.zeros((batch_size, 5)) # [time, quat]
            virtual_data[:,0] = times[:,j]
            virtual_data[:,1:] = out
            virtual_data = np.expand_dims(virtual_data, axis = 1)

            if virtual_queue is None:
                virtual_queue = virtual_data
            else:
                virtual_queue = np.concatenate((virtual_queue, virtual_data), axis = 1)
    return avg_loss.avg, np.squeeze(virtual_queue, axis=0)

def inference(cf, model, data_path, USE_CUDA):
    print("-----------Load Dataset----------")
    test_loader = get_inference_data_loader(cf, data_path)

    start_time = time.time()
    test_loss, virtual_queue = run(model, test_loader, cf, USE_CUDA=USE_CUDA)
    time_used = (time.time() - start_time) / 60
    print("TestLoss: %.4f | Time_used: %.4f minutes" % (test_loss, time_used))
    
    video_name = data_path.split("/")[-1]
    virtual_path = os.path.join("./test", cf['data']['exp'], video_name+'.txt')
    np.savetxt(virtual_path, virtual_queue, delimiter=' ')
    # virtual_queue = np.loadtxt(virtual_path)

    data = test_loader.dataset.data[0]

    print("------Start Visual Result--------")
    rotations_real, lens_offsets_real = get_rotations(data.frame, data.gyro, data.ois, data.length)
    rotations_virtual, lens_offsets_virtual = get_rotations(data.frame, virtual_queue, np.zeros(data.ois.shape), data.length)

    path = os.path.join("./test", cf['data']['exp'], video_name+'.jpg')
    visual_rotation(rotations_real, rotations_virtual, lens_offsets_real, lens_offsets_virtual, path)

    print("------Start Warping Video--------")
    grid = get_grid(data.frame, data.gyro, data.ois, virtual_queue[:,1:])
    print(grid.shape)
    video_path = os.path.join(data_path, video_name+".mp4")
    save_path = os.path.join("./test", cf['data']['exp'], video_name+'_stab.mp4')
    warp_video(grid, video_path, save_path)
    return

def main(args = None):
    config_file = args.config
    dir_path = args.dir_path
    cf = yaml.load(open(config_file, 'r'))
    
    USE_CUDA = cf['data']["use_cuda"]

    checkpoints_dir = cf['data']['checkpoints_dir']
    checkpoints_dir = make_dir(checkpoints_dir, cf)

    log_file = open(os.path.join(cf["data"]["log"], cf['data']['exp']+'_test.log'), 'w+')
    printer = Printer(sys.stdout, log_file).open()

    # Define the model
    model = Model(cf) 

    print("------Load Pretrined Model-------")
    checkpoint = torch.load(os.path.join(checkpoints_dir, cf['data']['exp']+'_last.checkpoint'))
    model.net.load_state_dict(checkpoint['state_dict'])
                
    if USE_CUDA:
        model.net.cuda()

    data_name = sorted(os.listdir(dir_path))
    for i in range(len(data_name)):
        print("Running Inference: " + str(i+1) + "/" + str(len(data_name)))
        inference(cf, model, os.path.join(dir_path, data_name[i]), USE_CUDA)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--config", default="./conf/sample.yaml", help="Config file.")
    parser.add_argument("--dir_path", default="/home/zhmeishi_google_com/dataset/Google/test/")
    args = parser.parse_args()
    main(args = args)