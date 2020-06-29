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
from dataset import get_data_loader, get_virtual_data
from model import Model
import datetime
import copy
from util import make_dir, get_optimizer, AverageMeter, save_train_info

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

            if is_training:
                out = model.net(inputs)
            else:
                with torch.no_grad():
                    out = model.net(inputs)
        
            loss = model.loss(out, virtual_inputs, real_inputs_step)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if clip_norm:
                    nn.utils.clip_grad_norm_(model.net.parameters(), max_norm=clip_norm)
                optimizer.step()

            avg_loss.update(loss.item(), batch_size) 
            
            if USE_CUDA:
                out = out.cpu().detach().numpy()

            virtual_data = np.zeros((batch_size, 5))
            virtual_data[:,0] = times[:,j]
            virtual_data[:,1:] = out
            virtual_data = np.expand_dims(virtual_data, axis = 1)

            if virtual_queue is None:
                virtual_queue = virtual_data
            else:
                virtual_queue = np.concatenate((virtual_queue, virtual_data), axis = 1)
    
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

    print("-----------Load Dataset----------")
    train_loader, test_loader = get_data_loader(cf)

    for idx, m in enumerate(model.net.children()):
        print('{}:{}'.format(idx,m))
                
    if USE_CUDA:
        model.net.cuda()

    print("----------Start Training----------")
    currentDT = datetime.datetime.now()
    print(currentDT.strftime(" %Y-%m-%d %H:%M:%S"))
    
    start_time = time.time()

    if lr_step:
        decay_epoch = list(range(1+lr_step, epochs+1, lr_step))
    
    lr = init_lr

    for count in range(1, epochs+1):
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