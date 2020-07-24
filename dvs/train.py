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
from dataset import get_data_loader
from model import Model
import datetime
import copy
from util import make_dir, get_optimizer, AverageMeter, save_train_info, norm_flow

def run_epoch(model, loader, cf, epoch, lr, optimizer=None, is_training=True, USE_CUDA=True, clip_norm=0):
    number_virtual, number_real = cf['data']["number_virtual"], cf['data']["number_real"]
    sample_freq = cf['data']["sample_freq"]
    avg_loss = AverageMeter()
    if is_training:
        model.net.train()
    else:
        model.net.eval()
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real_inputs, times, flo, flo_back, real_projections, real_postion, real_queue_idx = data
        print("Fininsh Load data")

        real_inputs = real_inputs.type(torch.float) #[b,60,84=21*4]
        real_projections = real_projections.type(torch.float) 
        flo = flo.type(torch.float) 
        flo_back = flo_back.type(torch.float) 

        batch_size, step, dim = real_inputs.size()
        times = times.numpy()
        real_queue_idx = real_queue_idx.numpy()
        virtual_queue = [None] * batch_size
        loss = 0
        model.net.init_hidden(batch_size)
        for j in range(step):
            if (j+1) % 10 == 0:
                print("Step: "+str(j+1)+"/"+str(step))
            virtual_inputs, vt_1 = loader.dataset.get_virtual_data(virtual_queue, real_queue_idx, times[:, j], times[:, j+1], times[:, 0], batch_size, number_virtual, sample_freq) # [b,40=4*10]
            
            real_inputs_step = real_inputs[:,j,:]
            # inputs = torch.cat((real_inputs_step,virtual_inputs), dim = 1) 
            inputs = Variable(real_inputs_step)
            if USE_CUDA:
                real_inputs_step = real_inputs_step.cuda()
                virtual_inputs = virtual_inputs.cuda()
                inputs = inputs.cuda()
                flo_step = flo[:,j].cuda()
                flo_back_step = flo_back[:,j].cuda()
                vt_1 = vt_1.cuda()
                real_projections_t = real_projections[:,j+1].cuda()
                real_projections_t_1 = real_projections[:,j].cuda()
                real_postion_step = real_postion[:,j].cuda()

            b, h, w, _ = flo_step.size()
            flo_step = norm_flow(flo_step, h, w)
            flo_back_step = norm_flow(flo_back_step, h, w)

            if is_training:
                out = model.net(inputs)
            else:
                with torch.no_grad():
                    out = model.net(inputs)

            if j < step - 4 and j > 3:
                follow = False
            else:
                follow = True

            if epoch > 50:
                undefine = True
            else:
                undefine = False
            loss += model.loss(out, vt_1, virtual_inputs, real_inputs_step, \
                flo_step, flo_back_step, real_projections_t, real_projections_t_1, real_postion_step, \
                follow = follow, undefine = undefine)
            
            if USE_CUDA:
                out = out.cpu().detach().numpy() 
                real_postion_step = real_postion_step.cpu().numpy()
                # print(j)
                # print(inputs.cpu().detach().numpy())
                # print(out)
                # print(real_inputs_step[:,40:44].cpu().detach().numpy())

            virtual_queue = loader.dataset.update_virtual_queue(batch_size, virtual_queue, out, times[:,j+1], real_postion_step)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            if clip_norm:
                nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=clip_norm)
            optimizer.step()

        avg_loss.update(loss.item(), batch_size) 
    
    return avg_loss.avg


def train(args = None):
    torch.autograd.set_detect_anomaly(True)
    config_file = args.config
    cf = yaml.load(open(config_file, 'r'))
    
    USE_CUDA = cf['data']["use_cuda"]
    seed = cf['train']["seed"]
    
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)

    checkpoints_dir = cf['data']['checkpoints_dir']
    epochs = cf["train"]["epoch"]
    snapshot = cf["train"]["snapshot"]
    decay_epoch = cf['train']['decay_epoch']
    init_lr = cf["train"]["init_lr"]
    lr_decay = cf["train"]["lr_decay"]
    lr_step = cf["train"]["lr_step"]
    clip_norm = cf["train"]["clip_norm"]
    load_model = cf["model"]["load_model"]

    checkpoints_dir = make_dir(checkpoints_dir, cf)

    log_file = open(os.path.join(cf["data"]["log"], cf['data']['exp']+'.log'), 'w+')
    printer = Printer(sys.stdout, log_file).open()
    
    print('----Print Arguments Setting------') 
    for key in cf:
        print('{}:'.format(key))
        for para in cf[key]:
            print('{:50}:{}'.format(para,cf[key][para]))
        print('\n')

    # Define the model
    model = Model(cf) 
    optimizer = get_optimizer(cf["train"]["optimizer"], model, init_lr, cf)

    for idx, m in enumerate(model.net.children()):
        print('{}:{}'.format(idx,m))

    if load_model is not None:
        print("------Load Pretrined Model--------")
        checkpoint = torch.load(load_model)
        model.net.load_state_dict(checkpoint['state_dict'])
        print("------Resume Training Process-----")
        epoch_load = checkpoint['epoch']
        print("Epoch load: ", epoch_load)
    else:
        epoch_load = 1
                
    if USE_CUDA:
        model.net.cuda()

    print("-----------Load Dataset----------")
    train_loader, test_loader = get_data_loader(cf)

    print("----------Start Training----------")
    currentDT = datetime.datetime.now()
    print(currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    
    start_time = time.time()

    if lr_step:
        decay_epoch = list(range(1+lr_step, epochs+1, lr_step))
    
    lr = init_lr

    for count in range(epoch_load, epochs+1):
        if decay_epoch != None and count in decay_epoch:
            lr *= lr_decay
            for param in optimizer.param_groups:
                param['lr'] *= lr_decay
        
        print("Epoch: %d, learning_rate: %.5f" % (count, lr))

        train_loss = run_epoch(model, train_loader, cf, count, lr, optimizer=optimizer, clip_norm=clip_norm, is_training=True, USE_CUDA=USE_CUDA)

        test_loss = run_epoch(model, test_loader, cf, count, lr, is_training=False, USE_CUDA=USE_CUDA)

        time_used = (time.time() - start_time) / 60
        print("Epoch %d done | TrLoss: %.4f | TestLoss: %.4f | Time_used: %.4f minutes" % (
            count, train_loss,  test_loss, time_used))

        if count % snapshot == 0:
            save_train_info("epoch", checkpoints_dir, cf, model, count, optimizer)
            save_train_info("last", checkpoints_dir, cf, model, count, optimizer)
            print("Model stored at epoch %d"%count)

    currentDT = datetime.datetime.now()
    print(currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    print("------------End Training----------")
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--config", default="./conf/sample.yaml", help="Config file.")
    args = parser.parse_args()
    train(args = args)