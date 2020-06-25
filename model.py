import math
import torch
from collections import OrderedDict

import torch.nn as nn
import numpy as np
import util
import yaml
import os
from loss import C2_Smooth_loss

Activates = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}

class LayerCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, 
                        activation_function=nn.ReLU, batch_norm=True):
        super(LayerCNN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)
        if pooling_size is not None:
            self.pooling = nn.MaxPool2d(pooling_size)
        else:
            self.pooling = None
        
    def forward(self, x):
        x = self.conv(x)     #x->[batch,channel,height,width]
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x

class LayerFC(nn.Module):
    def __init__(self, in_features, out_features, bias, drop_out=0, activation_function=nn.ReLU, batch_norm = False):
        super(LayerFC, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation_function(inplace=True) if activation_function is not None else None
        self.dropout = nn.Dropout(p=drop_out,inplace=False) if drop_out else None
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Net(nn.Module):
    def __init__(self, cf):
        super(Net, self).__init__()
        self.cnn_param = cf["model"]["cnn"]
        self.fc_param = cf["model"]["fc"]

        self._fc_input_size = (2*cf["data"]["number_real"]+1+cf["data"]["number_virtual"]) * 4

        #CNN Layers
        cnns = []
        cnn_activation = Activates[self.cnn_param["activate_function"]]
        cnn_batch_norm = self.cnn_param["batch_norm"]
        cnn_layer_param = self.cnn_param["layers"]
        if cnn_layer_param is not None:
            cnn_layers = len(cnn_layer_param)
            for layer in range(cnn_layers):
                in_channel = eval(cnn_layer_param[layer][0])[0]
                out_channel = eval(cnn_layer_param[layer][0])[1]
                kernel_size = eval(cnn_layer_param[layer][1])
                stride = eval(cnn_layer_param[layer][2])
                padding = eval(cnn_layer_param[layer][3])
                pooling_size = eval(cnn_layer_param[layer][4])

                cnn = None
                cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, pooling_size, 
                            activation_function=cnn_activation, batch_norm=cnn_batch_norm)
                cnns.append(('%d' % layer, cnn))
        
                self._fc_input_size = int(math.floor((self._fc_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)
                if pooling_size is not None:
                    self._fc_input_size = int(math.floor((self._fc_input_size-pooling_size[1])/pooling_size[1])+1)
            self.convs = nn.Sequential(OrderedDict(cnns))

        else:
            self.convs = None
            out_channel = cf["data"]["channel_size"]
            
        self.gap = nn.AvgPool2d(self._fc_input_size) if self.cnn_param["gap"] else None
        self._fc_input_size = out_channel if self.cnn_param["gap"] else out_channel*(self._fc_input_size)

        #FC Layers
        fcs = []
        fc_activation = Activates[self.fc_param["activate_function"]]
        fc_batch_norm = self.fc_param["batch_norm"]
        fc_layer_param = self.fc_param["layers"]
        fc_drop_out = self.fc_param["drop_out"]
        fc_layers = len(fc_layer_param)
        
        if fc_layers == 1:
            fc = LayerFC(self._fc_input_size,fc_layer_param[0][0],fc_layer_param[0][1],
                    fc_drop_out, None, fc_batch_norm)
            fcs.append(('%d'%(fc_layers-1), fc))
        else:
            for layer in range(fc_layers-1):
                if layer:
                    fc = LayerFC(fc_layer_param[layer-1][0],fc_layer_param[layer][0],fc_layer_param[layer][1],
                        fc_drop_out, fc_activation, fc_batch_norm)
                else:
                    fc = LayerFC(self._fc_input_size,fc_layer_param[layer][0],fc_layer_param[layer][1],
                        fc_drop_out,fc_activation, fc_batch_norm)
                fcs.append(('%d'%layer, fc))
            fc = LayerFC(fc_layer_param[fc_layers-2][0],fc_layer_param[fc_layers-1][0],fc_layer_param[fc_layers-1][1],
                        fc_drop_out,None, fc_batch_norm)
            fcs.append(('%d'%(fc_layers-1), fc))

        self.class_num = fc_layer_param[fc_layers-1][0]
        self.fcs = nn.Sequential(OrderedDict(fcs))

    def forward(self, x):
        b,c = x.size()   #x->[batch,channel,height,width]
        if self.convs is not None:
            x = self.convs(x)
        if self.gap is not None:
            x = self.gap(x)
        x = x.view(b,-1)
        x = self.fcs(x) # [b, 4]
        x = x / (torch.unsqueeze(torch.norm(x, dim = 1), 1) + 1e-6)
        return x

class Model():
    def __init__(self, cf):
        super().__init__()
        self.net = Net(cf)
        self.init_weights(cf)
        
        self.loss_smooth = nn.MSELoss()
        self.loss_follow = nn.MSELoss()
        self.loss_c2_smooth = C2_Smooth_loss()

    def loss(self, out, virtual_inputs, real_inputs):
        return self.loss_smooth(out, virtual_inputs[:, :4]) + self.loss_follow(out, real_inputs) \
            + self.loss_c2_smooth(out,virtual_inputs[:, :4], virtual_inputs[:, 4:8])

    def init_weights(self, cf):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or  isinstance(m, nn.Linear):
                if cf["train"]["init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data)
                elif cf["train"]["init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data)

    def save_checkpoint(self, epoch = 0, optimizer=None):
        package = {
                'cnn': self.net.cnn_param,
                'fc': self.net.fc_param,
                'state_dict': self.net.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        package["epoch"] = epoch
        return package