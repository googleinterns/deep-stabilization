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
from util import make_dir, get_optimizer, norm_flow
from gyro import (
    get_grid, 
    get_rotations, 
    visual_rotation,
    torch_QuaternionProduct,
    torch_norm_quat
    )
from warp import warp_video

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(model, loader, cf, USE_CUDA=True):
    no_flo = False
    number_virtual, number_real = cf['data']["number_virtual"], cf['data']["number_real"]
    model.net.eval()
    model.unet.eval()
    activation = nn.Softshrink(0.0006) # 0.0036
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

        run_loss = 0
        model.net.init_hidden(batch_size)
        count = 0
        for j in range(step):
            if (j+1) % 100 == 0:
                print("Step: "+str(j+1)+"/"+str(step))
            virtual_inputs, vt_1 = loader.dataset.get_virtual_data(
                virtual_queue, real_queue_idx, times[:, j], times[:, j+1], times[:, 0], batch_size, number_virtual, real_postion[:,j]) 
            real_inputs_step = real_inputs[:,j,:]
            inputs = torch.cat((real_inputs_step,virtual_inputs), dim = 1) 

            # inputs = Variable(real_inputs_step)
            if USE_CUDA:
                real_inputs_step = real_inputs_step.cuda()
                virtual_inputs = virtual_inputs.cuda()
                inputs = inputs.cuda()
                if no_flo is False:
                    flo_step = flo[:,j].cuda()
                    flo_back_step = flo_back[:,j].cuda()
                else:
                    flo_step = None
                    flo_back_step = None
                vt_1 = vt_1.cuda()
                real_projections_t = real_projections[:,j+1].cuda()
                real_projections_t_1 = real_projections[:,j].cuda()
                real_postion_anchor = real_postion[:,j].cuda()
                ois_step = ois[:,j].cuda()

            if no_flo is False:
                b, h, w, _ = flo_step.size()
                flo_step = norm_flow(flo_step, h, w)
                flo_back_step = norm_flow(flo_back_step, h, w)

            with torch.no_grad():
                if no_flo is False:
                    flo_out = model.unet(flo_step, flo_back_step)
                else:
                    flo_out = None
                if j < 1:
                    for i in range(2):
                        out = model.net(inputs, flo_out, ois_step)
                else:
                    out = model.net(inputs, flo_out, ois_step)

            real_position = real_inputs_step[:,40:44]
            virtual_position = virtual_inputs[:, -4:]

            out[:, :3] = activation(out[:, :3])
            out = torch_norm_quat(out)

            pos = torch_QuaternionProduct(virtual_position, real_postion_anchor)
            loss_step = model.loss(out, vt_1, virtual_inputs, real_inputs_step, \
                flo_step, flo_back_step, real_projections_t, real_projections_t_1, real_postion_anchor, \
                follow = True, optical = True, undefine = True)
            run_loss += loss_step

            out = torch_QuaternionProduct(out, pos)

            if USE_CUDA:
                out = out.cpu().detach().numpy() 

            virtual_queue = loader.dataset.update_virtual_queue(batch_size, virtual_queue, out, times[:,j+1])
    
    run_loss /= step
    print( "\nLoss: follow, angle, smooth, c2_smooth, undefine, optical")
    print(run_loss.cpu().numpy()[:-1], "\n")
    return np.squeeze(virtual_queue, axis=0)


def inference(cf, data_path, USE_CUDA):
    checkpoints_dir = cf['data']['checkpoints_dir']
    checkpoints_dir = make_dir(checkpoints_dir, cf)
    files = os.listdir(data_path)
    for f in files:
        if f[-3:] == "mp4" and "no_ois" not in f and "gimbal" not in f.lower() and "grid" not in f.lower() and "flo" not in f.lower():
            video_name = f[:-4]

    # Define the model
    model = Model(cf) 
    load_model = cf["model"]["load_model"]

    print("------Load Pretrined Model--------")
    if load_model is not None:
        checkpoint = torch.load(load_model)
        print(load_model)
    else:
        load_last = os.path.join(checkpoints_dir, cf['data']['exp']+'_last.checkpoint')
        checkpoint = torch.load(load_last)
        print(load_last)
    model.net.load_state_dict(checkpoint['state_dict'])
    model.unet.load_state_dict(checkpoint['unet'])
                
    if USE_CUDA:
        model.net.cuda()
        model.unet.cuda()

    print("-----------Load Dataset----------")
    test_loader = get_inference_data_loader(cf, data_path, no_flo = False)
    data = test_loader.dataset.data[0]

    start_time = time.time()
    virtual_queue= run(model, test_loader, cf, USE_CUDA=USE_CUDA)

    virtual_data = np.zeros((1,5))
    virtual_data[:,1:] = virtual_queue[0, 1:]
    virtual_data[:,0] = data.frame[0,0]
    virtual_queue = np.concatenate((virtual_data, virtual_queue), axis = 0)

    print(virtual_queue.shape)
    time_used = (time.time() - start_time) / 60

    print("Time_used: %.4f minutes" % (time_used))

    
    virtual_path = os.path.join("./test", cf['data']['exp'], data_path.split("/")[-1]+'.txt')
    np.savetxt(virtual_path, virtual_queue, delimiter=' ')

    print("------Start Warping Video--------")
    grid = get_grid(test_loader.dataset.static_options, \
        data.frame[:data.length], data.gyro, data.ois, virtual_queue[:data.length,1:], no_shutter = False)
    return data, virtual_queue, video_name, grid

def visual_result(cf, data, video_name, virtual_queue, virtual_queue2 = None, compare_exp = None):
    print("------Start Visual Result--------")
    rotations_virtual, lens_offsets_virtual = get_rotations(data.frame[:data.length], virtual_queue, np.zeros(data.ois.shape), data.length)
    rotations_real, lens_offsets_real = get_rotations(data.frame[:data.length], data.gyro, data.ois, data.length)
    if virtual_queue2 is not None:
        rotations_virtual2, lens_offsets_virtual2 = get_rotations(data.frame[:data.length], virtual_queue2, np.zeros(data.ois.shape), data.length)
        path = os.path.join("./test", cf['data']['exp'], video_name+'_'+compare_exp+'.jpg')
    else:
        rotations_virtual2, lens_offsets_virtual2 = None, None
        path = os.path.join("./test", cf['data']['exp'], video_name+'.jpg')
    
    visual_rotation(rotations_real, lens_offsets_real, rotations_virtual, lens_offsets_virtual, rotations_virtual2, lens_offsets_virtual2, path)


def main(args = None):
    config_file = args.config
    dir_path = args.dir_path
    cf = yaml.load(open(config_file, 'r'))

    USE_CUDA = cf['data']["use_cuda"]

    log_file = open(os.path.join(cf["data"]["log"], cf['data']['exp']+'_test.log'), 'w+')
    printer = Printer(sys.stdout, log_file).open()

    data_name = sorted(os.listdir(dir_path))
    for i in range(len(data_name)):
        print("Running Inference: " + str(i+1) + "/" + str(len(data_name)))
        save_path = os.path.join("./test", cf['data']['exp'], data_name[i]+'_stab.mp4')

        data_path = os.path.join(dir_path, data_name[i])
        data, virtual_queue, video_name, grid= inference(cf, data_path, USE_CUDA)

        virtual_queue2 = None
        visual_result(cf, data, data_name[i], virtual_queue, virtual_queue2 = virtual_queue2, compare_exp = None)

        video_path = os.path.join(data_path, video_name+".mp4")
        warp_video(grid, video_path, save_path, frame_number = False)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--config", default="./conf/iccv_6.yaml", help="Config file.")
    parser.add_argument("--dir_path", default="./video")
    args = parser.parse_args()
    main(args = args)