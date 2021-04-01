import math
import torch
from collections import OrderedDict

import torch.nn as nn
import numpy as np
import util
import yaml
import os
from loss import C2_Smooth_loss, C1_Smooth_loss, Optical_loss, Undefine_loss, Angle_loss, Follow_loss, Stay_loss
from gyro import torch_norm_quat, torch_QuaternionProduct
import torch.nn.functional as F

Activates = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}

class LayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super(LayerLSTM, self).__init__()
        self.LSTM = nn.LSTMCell(input_size, hidden_size, bias)
        self.hidden_size = hidden_size
    
    def init_hidden(self, batch_size):
        self.hx = torch.zeros((batch_size, self.hidden_size)).cuda()
        self.cx = torch.zeros((batch_size, self.hidden_size)).cuda()

    def forward(self, x):
        self.hx, self.cx = self.LSTM(x, (self.hx, self.cx))
        return self.hx
        

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
        # self.activation = activation_function(inplace=True) if activation_function is not None else None
        self.activation = activation_function() if activation_function is not None else None
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
        self.rnn_param = cf["model"]["rnn"]
        self.fc_param = cf["model"]["fc"]
        self.unit_size = 4
        self.no_flo = False

        if self.no_flo is False:
            self._rnn_input_size = (2*cf["data"]["number_real"]+1+cf["data"]["number_virtual"]) * 4 + 64
        else:
            self._rnn_input_size = (2*cf["data"]["number_real"]+1+cf["data"]["number_virtual"]) * self.unit_size

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
        
                self._rnn_input_size = int(math.floor((self._rnn_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)
                if pooling_size is not None:
                    self._rnn_input_size = int(math.floor((self._rnn_input_size-pooling_size[1])/pooling_size[1])+1)
            self.convs = nn.Sequential(OrderedDict(cnns))

        else:
            self.convs = None
            out_channel = cf["data"]["channel_size"]
            
        self.gap = nn.AvgPool2d(self._rnn_input_size) if self.cnn_param["gap"] else None
        self._rnn_input_size = out_channel if self.cnn_param["gap"] else out_channel*(self._rnn_input_size)

        #RNN Layers
        rnns = []
        rnn_layer_param = self.rnn_param["layers"]
        rnn_layers = len(rnn_layer_param)
        
        for layer in range(rnn_layers):
            if layer:
                rnn = LayerLSTM(rnn_layer_param[layer-1][0], rnn_layer_param[layer][0], rnn_layer_param[layer][1])
            else:
                rnn = LayerLSTM(self._rnn_input_size, rnn_layer_param[layer][0], rnn_layer_param[layer][1])
            rnns.append(('%d'%layer, rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        self._fc_input_size = rnn_layer_param[rnn_layers-1][0] #* 2 # ois
        
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
                        fc_drop_out,None, fc_batch_norm) # Modified
            fcs.append(('%d'%(fc_layers-1), fc))

        self.class_num = fc_layer_param[fc_layers-1][0]
        self.fcs = nn.Sequential(OrderedDict(fcs))

    def init_hidden(self, batch_size):
        for i in range(len(self.rnns)):
            self.rnns[i].init_hidden(batch_size)

    def forward(self, x, flo, ois):
        b,c = x.size()   #x->[batch,channel,height,width]
        if self.convs is not None:
            x = self.convs(x)
        if self.gap is not None:
            x = self.gap(x)
        x = x.view(b,-1)
        if self.no_flo is False:
            x = torch.cat((x, flo), dim = 1) 
        x = self.rnns(x)
        x = self.fcs(x) # [b, 4]
        x = torch_norm_quat(x)
        return x

class Model():
    def __init__(self, cf):
        super().__init__()
        self.net = Net(cf)
        self.unet = UNet()
        self.init_weights(cf)
        
        self.loss_smooth = C1_Smooth_loss()
        self.loss_follow = Follow_loss()
        self.loss_c2_smooth = C2_Smooth_loss()
        self.loss_optical = Optical_loss()
        self.loss_undefine = Undefine_loss(ratio = 0.08)
        self.loss_angle = Angle_loss()
        self.loss_stay = Stay_loss()

        self.loss_smooth_w = cf["loss"]["smooth"]
        self.loss_angle_w = cf["loss"]["angle"]
        self.loss_follow_w = cf["loss"]["follow"]
        self.loss_c2_smooth_w = cf["loss"]["c2_smooth"]
        self.loss_undefine_w = cf["loss"]["undefine"]
        self.loss_opt_w = cf["loss"]["opt"]
        self.loss_stay_w = cf["loss"]["stay"]

        self.gaussian_weight = np.array([0.072254, 0.071257, 0.068349, 0.063764, 0.057856, 0.051058, 0.043824, 0.036585, 0.029705, 0.023457, 0.01801])

    def loss(
        self, out, vt_1, virtual_inputs, real_inputs, flo, flo_back, 
        real_projections_t, real_projections_t_1, real_postion_anchor, 
        follow = True, undefine = True, optical = True, stay = False
        ):
        unit_size = self.net.unit_size
        mid = real_inputs.size()[1]//(2*unit_size) 

        Rt = real_inputs[:,unit_size*(mid):unit_size*(mid)+4] 
        v_pos = torch_QuaternionProduct(out, virtual_inputs[:, -4:])
        r_pos = torch_QuaternionProduct(v_pos, real_postion_anchor)

        loss = torch.zeros(7).cuda()
        if self.loss_follow_w > 0 and follow:
            for i in range(-2,3):
                loss[0] += self.loss_follow_w * self.loss_follow(v_pos, real_inputs[:,unit_size*(i+mid):unit_size*(i+mid)+4], None)
        if self.loss_angle_w > 0 and follow:
            threshold = 6 / 180 * 3.1415926
            loss_angle, theta = self.loss_angle(v_pos, Rt, threshold = threshold)
            loss[1] = self.loss_angle_w * loss_angle
        if self.loss_smooth_w > 0:
            loss_smooth = self.loss_smooth(out)
            loss[2] = self.loss_smooth_w * loss_smooth
        if self.loss_c2_smooth_w > 0: 
            loss[3] = self.loss_c2_smooth_w * self.loss_c2_smooth(out, virtual_inputs[:, -4:], virtual_inputs[:, -8:-4])
        if self.loss_undefine_w > 0 and undefine:
            Vt_undefine = v_pos.clone() 
            for i in range(0, 10, 2):
                Rt_undefine = real_inputs[:,unit_size*(mid+i):unit_size*(mid+i)+4]
                loss_undefine_w = self.loss_undefine_w * self.gaussian_weight[i]
                loss[4] +=  loss_undefine_w * self.loss_undefine(Vt_undefine, Rt_undefine)
                Vt_undefine = torch_QuaternionProduct(out, Vt_undefine)
                Vt_undefine = torch_QuaternionProduct(out, Vt_undefine)
        if self.loss_opt_w > 0 and optical:
            loss[5] = self.loss_opt_w * self.loss_optical(r_pos, vt_1, flo, flo_back, real_projections_t, real_projections_t_1) 
        if self.loss_stay_w > 0 and stay:
            loss[6] = self.loss_stay_w * self.loss_stay(out) 
        return loss


    def init_weights(self, cf):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or  isinstance(m, nn.Linear):
                if cf["train"]["init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data)
                elif cf["train"]["init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data)

        for m in self.unet.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or  isinstance(m, nn.Linear):
                if cf["train"]["init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight.data)
                elif cf["train"]["init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight.data)

    def save_checkpoint(self, epoch = 0, optimizer=None):
        package = {
                'cnn': self.net.cnn_param,
                'fc': self.net.fc_param,
                'state_dict': self.net.state_dict(),
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if self.unet is not None:
            package['unet'] = self.unet.state_dict()
        package["epoch"] = epoch
        return package


class UNet(nn.Module):
    def __init__(self, n_channels = 4, n_classes = 16, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        # factor = 2 if bilinear else 1
        self.down4 = Down(64, 128)
        self._fc_input_size = 128 * 1 * 1
        self.fc = LayerFC(self._fc_input_size, 64, bias = True)

    def forward(self, x, x_back = None):
        if x_back is not None:
            x = torch.cat((x,x_back), dim =3)
        x = x.permute(0,3,1,2)
        b,c,h,w = x.size()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = torch.reshape(x5, (b, -1))
        x = self.fc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)