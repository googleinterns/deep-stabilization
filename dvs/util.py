import os
import torch
import cv2

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
        optimizer = torch.optim.Adam(model.net.parameters(), lr=init_lr, weight_decay=cf["train"]["weight_decay"])
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.net.parameters(), lr=init_lr, momentum=cf["train"]["momentum"])
    return optimizer

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

def norm_flow(flow, h, w):
    flow[:,:,:,0] /= h
    flow[:,:,:,1] /= w
    return flow
