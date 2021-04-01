#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, datasets
from utils import flow_utils, tools
import time

    # Reusable function for inference
def inference(args, epoch, data_path, data_loader, model, offset=0):

    model.eval()
    
    if args.save_flow or args.render_validation:
        flow_folder = "{}/flo".format(data_path)
        flow_back_folder = "{}/flo_back".format(data_path)
        if not os.path.exists(flow_folder):
            os.makedirs(flow_folder)
        if not os.path.exists(flow_back_folder):
            os.makedirs(flow_back_folder)
    
    # visualization folder
    if args.inference_visualize:
        flow_vis_folder = "{}/flo_vis".format(data_path)
        if not os.path.exists(flow_vis_folder):
            os.makedirs(flow_vis_folder)
        flow_back_vis_folder = "{}/flo_back_vis".format(data_path)
        if not os.path.exists(flow_back_vis_folder):
            os.makedirs(flow_back_vis_folder)
    
    args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

    progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ', 
        leave=True, position=offset)

    for batch_idx, (data) in enumerate(progress):
        data = data[0]
        data_back = torch.cat((data[:,:,1:,:,:], data[:,:,:1,:,:]), dim = 2)
        if args.cuda:
            data_forward = data.cuda(non_blocking=True)
            data_back = data_back.cuda(non_blocking=True)
        data_forward = Variable(data_forward)
        data_back = Variable(data_back)

        flo_path = join(flow_folder, '%06d.flo'%(batch_idx))
        flo_back_path = join(flow_back_folder, '%06d.flo'%(batch_idx))
        frame_size = data_loader.dataset.frame_size
        if not os.path.exists(flo_path):
            with torch.no_grad():
                output = model(data_forward)[:,:,:frame_size[0], :frame_size[1]]
            if args.save_flow or args.render_validation:
                _pflow = output[0].data.cpu().numpy().transpose(1, 2, 0)
                flow_utils.writeFlow( flo_path,  _pflow)
                if args.inference_visualize:
                    flow_utils.visulize_flow_file(
                        join(flow_folder, '%06d.flo' % (batch_idx)),flow_vis_folder)

        if not os.path.exists(flo_back_path):
            with torch.no_grad():
                output = model(data_back)[:,:,:frame_size[0], :frame_size[1]]
            if args.save_flow or args.render_validation:
                _pflow = output[0].data.cpu().numpy().transpose(1, 2, 0)
                flow_utils.writeFlow( flo_back_path,  _pflow)
                if args.inference_visualize:
                    flow_utils.visulize_flow_file(
                        join(flow_back_folder, '%06d.flo' % (batch_idx)), flow_back_vis_folder)
                
        progress.update(1)

        if batch_idx == (args.inference_n_batches - 1):
            break
    progress.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument("--rgb_max", type=float, default = 255.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--save', '-s', default='./Google', type=str, help='directory for saving')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_visualize', action='store_true',
                        help="visualize the optical flow during inference")
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='Google', 
                                    skip_params=['is_cropped'],
                                    parameter_defaults={'root': './Google/train',
                                                        'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]

        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        args.total_epochs = 1
        args.inference_dir = "{}/inference".format(args.save)

    print('Source Code')
    # print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers, 
                   'pin_memory': True, 
                   'drop_last' : True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        block.log('Inference Dataset: {}'.format(args.inference_dataset))

        dataset_root = args.inference_dataset_root 
        data_name = sorted(os.listdir(dataset_root))

        block.log(data_name)
        inference_loaders = {}
        for i in range(len(data_name)):
            dataset_path = os.path.join(dataset_root, data_name[i])
            args.inference_dataset_root  = dataset_path
            inference_dataset = args.inference_dataset_class(args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
            inference_loaders[dataset_path] = DataLoader(inference_dataset, batch_size=args.effective_inference_batch_size, shuffle=False, **inf_gpuargs)
            block.log('Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class Model(nn.Module):
            def __init__(self, args):
                super(Model, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                
            def forward(self, data):
                output = self.model(data)
                return output

        model = Model(args)

        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        if args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model = model.cuda()
            block.log('Parallelizing')
            model = nn.parallel.DataParallel(model, device_ids=list(range(args.number_gpus)))

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()

        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # Log all arguments to file
    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))

    for data_path in inference_loaders:
        # Primary epoch loop
        progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100, desc='Overall Progress', leave=True, position=0)
        offset = 1

        for epoch in progress:
            stats = inference(args=args, epoch=epoch - 1, data_path = data_path, data_loader=inference_loaders[data_path], model=model, offset=offset)
            offset += 1
        print("\n")