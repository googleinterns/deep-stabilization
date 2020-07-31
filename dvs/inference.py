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
from dataset import get_data_loader, get_inference_data_loader
from model import Model
import datetime
import copy
from util import make_dir, get_optimizer, AverageMeter, norm_flow
from gyro import (
    get_grid, 
    get_rotations, 
    visual_rotation, 
    GetGyroAtTimeStamp, 
    torch_ConvertQuaternionToAxisAngle, 
    torch_ConvertAxisAngleToQuaternion,
    torch_QuaternionProduct
    )
from warp import warp_video

def run(model, loader, cf, USE_CUDA=True):
    number_virtual, number_real = cf['data']["number_virtual"], cf['data']["number_real"]
    sample_freq = cf['data']["sample_freq"]
    avg_loss = AverageMeter()

    model.net.eval()
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real_inputs, times, flo, flo_back, real_projections, real_postion, ois, real_queue_idx = data
        print("Fininsh Load data")

        real_inputs = real_inputs.type(torch.float) #[b,60,84=21*4]
        real_projections = real_projections.type(torch.float) 
        flo = flo.type(torch.float) 
        flo_back = flo_back.type(torch.float) 
        ois = ois.type(torch.float)

        batch_size, step, dim = real_inputs.size()
        times = times.numpy()
        real_queue_idx = real_queue_idx.numpy()
        virtual_queue = [None] * batch_size
        losses = [0]
        model.net.init_hidden(batch_size)
        for j in range(step):
            # if j > 25:
            #     break
            if (j+1) % 10 == 0:
                print("Step: "+str(j+1)+"/"+str(step))
            virtual_inputs, vt_1 = loader.dataset.get_virtual_data(
                virtual_queue, real_queue_idx, times[:, j], times[:, j+1], times[:, 0], batch_size, number_virtual, real_postion[:,j], sample_freq) 
            real_inputs_step = real_inputs[:,j,:]
            inputs = torch.cat((real_inputs_step,virtual_inputs), dim = 1) 
            # inputs = Variable(real_inputs_step)
            if USE_CUDA:
                real_inputs_step = real_inputs_step.cuda()
                virtual_inputs = virtual_inputs.cuda()
                inputs = inputs.cuda()
                # flo_step = flo[:,j].cuda()
                # flo_back_step = flo_back[:,j].cuda()
                flo_step = None
                flo_back_step = None
                vt_1 = vt_1.cuda()
                real_projections_t = real_projections[:,j+1].cuda()
                real_projections_t_1 = real_projections[:,j].cuda()
                real_postion_step = real_postion[:,j].cuda()
                ois_step = ois[:,j].cuda()

            # b, h, w, _ = flo_step.size()
            # flo_step = norm_flow(flo_step, h, w)
            # flo_back_step = norm_flow(flo_back_step, h, w)

            # print(inputs)
            with torch.no_grad():
                # flo_out = model.unet(flo_step, flo_back_step)
                flo_out = None
                # out = model.net(inputs, flo_out, ois_step)
                # print("==")
                # print(out )
                # print(real_inputs_step[:,40:44])
                # print((real_postion_step))
                if j < 1:
                    for i in range(10):
                        out = model.net(inputs, flo_out, ois_step)
                else:
                    out = model.net(inputs, flo_out, ois_step)
                # print(inputs)
            # out[:,:3] *= 0
            # out[:,3] /= out[:,3]
            # real = real_inputs_step[:,40:44]

            # print("======")
            # print(virtual_inputs)
            # print(out)
            
            loss = model.loss(out, vt_1, virtual_inputs, real_inputs_step, flo_step, flo_back_step, real_projections_t, real_projections_t_1, real_postion_step, undefine = True)
            avg_loss.update(loss.item(), batch_size) 
            
            virtual_position = virtual_inputs[:, -4:]
            # if j % 50 > 100:
            out = torch_QuaternionProduct(out, virtual_position)
            out = torch_QuaternionProduct(out, real_postion_step) # [0.001, 0, 0, 1], [0, 0, 0, 1]
            if USE_CUDA:
                out = out.cpu().detach().numpy() 
                real = real_inputs_step[:,40:44].cpu().detach().numpy()
                # print(j)
                # print(inputs.cpu().detach().numpy())
                # print(out)
                # print(real)
                losses.append(loss.cpu().detach().numpy())
                # print(losses[-1])
            # print(out)

            virtual_queue = loader.dataset.update_virtual_queue(batch_size, virtual_queue, out, times[:,j+1])

    return avg_loss.avg, np.squeeze(virtual_queue, axis=0), losses


def inference(cf, model, data_path, USE_CUDA):
    print("-----------Load Dataset----------")
    test_loader = get_inference_data_loader(cf, data_path)
    data = test_loader.dataset.data[0]

    start_time = time.time()
    loss, virtual_queue, losses = run(model, test_loader, cf, USE_CUDA=USE_CUDA)

    virtual_data = np.zeros((1,5))
    # virtual_data[:,1:] = GetGyroAtTimeStamp(data.gyro, data.frame[0,0])
    virtual_data[:,1:] = virtual_queue[0, 1:]
    virtual_data[:,0] = data.frame[0,0]
    virtual_queue = np.concatenate((virtual_data, virtual_queue), axis = 0)

    # virtual_queue = np.concatenate((virtual_queue[:1], virtual_queue), axis = 0)
    # virtual_queue[0,0] = data.frame[0,0]

    print(virtual_queue.shape)
    time_used = (time.time() - start_time) / 60

    print("TestLoss: %.4f | Time_used: %.4f minutes" % (loss, time_used))
    
    video_name = data_path.split("/")[-1]
    virtual_path = os.path.join("./test", cf['data']['exp'], video_name+'.txt')
    np.savetxt(virtual_path, virtual_queue, delimiter=' ')
    # virtual_queue = np.loadtxt(virtual_path)

    print("------Start Visual Result--------")
    rotations_virtual, lens_offsets_virtual = get_rotations(data.frame[:data.length], virtual_queue, np.zeros(data.ois.shape), data.length)
    rotations_real, lens_offsets_real = get_rotations(data.frame[:data.length], data.gyro, data.ois, data.length)
    
    path = os.path.join("./test", cf['data']['exp'], video_name+'.jpg')
    visual_rotation(rotations_real, rotations_virtual, lens_offsets_real, lens_offsets_virtual, path)

    # data.length = 300
    print("------Start Warping Video--------")
    grid = get_grid(test_loader.dataset.static_options, \
        data.frame[:data.length], data.gyro, data.ois, virtual_queue[:data.length,1:])

    video_path = os.path.join(data_path, video_name+".mp4")
    save_path = os.path.join("./test", cf['data']['exp'], video_name+'_stab.mp4')
    warp_video(grid, video_path, save_path, losses = losses[:data.length])
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
        model.unet.cuda()

    data_name = sorted(os.listdir(dir_path))
    for i in range(len(data_name)):
        print("Running Inference: " + str(i+1) + "/" + str(len(data_name)))
        inference(cf, model, os.path.join(dir_path, data_name[i]), USE_CUDA)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--config", default="./conf/sample.yaml", help="Config file.")
    parser.add_argument("--dir_path", default="/mnt/disks/dataset/Google/test/")
    # parser.add_argument("--dir_path", default="/home/zhmeishi_google_com/dvs/data/testdata/videos_with_zero_virtual_motion/inputs_no_rotation")
    args = parser.parse_args()
    main(args = args)