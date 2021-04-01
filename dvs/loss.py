import torch
import numpy as np
from torch.autograd import Variable
import operator
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gyro import (
    torch_QuaternionProduct, 
    torch_QuaternionReciprocal, 
    get_static, 
    torch_GetVirtualProjection,
    torch_GetForwardGrid,
    torch_GetWarpingFlow,
    torch_ConvertAxisAngleToQuaternion,
    torch_ConvertQuaternionToAxisAngle,
    torch_norm_quat,
    torch_GetHomographyTransformFromProjections,
    torch_ApplyTransform
)
        
class C2_Smooth_loss(torch.nn.Module):
    def __init__(self):
        super(C2_Smooth_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, Qt, Qt_1, Qt_2):
        detaQt_1 = torch_QuaternionProduct(Qt_1, torch_QuaternionReciprocal(Qt_2))
        return self.MSE(Qt, detaQt_1)

class C1_Smooth_loss(torch.nn.Module):
    def __init__(self):
        super(C1_Smooth_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, v_r_axis, v_axis_t_1 = None, real_postion = None):
        quat_zero = torch.zeros(v_r_axis.shape).cuda()
        quat_zero[:,3] = 1
        return self.MSE(v_r_axis, quat_zero)

class Follow_loss(torch.nn.Module):
    def __init__(self):
        super(Follow_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, virtual_quat, real_quat, real_postion = None):
        if real_postion is not None:
            real_quat = torch_QuaternionProduct(real_quat, real_postion)
        return self.MSE(virtual_quat, real_quat)

class Stay_loss(torch.nn.Module):
    def __init__(self):
        super(Stay_loss, self).__init__()
        self.zero = torch.tensor([0.0,0.0,0.0,1.0]).cuda()

    def forward(self, virtual_quat):
        return torch.mean(torch.abs(virtual_quat - self.zero))


class Angle_loss(torch.nn.Module):
    def __init__(self):
        super(Angle_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, Q1, Q2, threshold = 0.5236, logistic_beta1 = 100):
        batch_size = Q1.shape[0]
        Q3 = torch_norm_quat(torch_QuaternionProduct(Q2, torch_QuaternionReciprocal(Q1)))
        theta = torch.zeros(batch_size).cuda()
        index = (Q3[:,3] < 1).nonzero()
        theta[index] = torch.acos(Q3[index,3]) * 2
        loss = torch.mean(theta * (1 / (1 + torch.exp(-logistic_beta1 * (theta - threshold)))))
        return loss, theta

class Optical_loss(torch.nn.Module):
    def __init__(self):
        super(Optical_loss, self).__init__()
        self.static_options = get_static()
        self.mesh = get_mesh()

    def forward(self, Vt, Vt_1, flo, flo_back, real_projection_t, real_projection_t_1):
        virtual_projection_t = torch_GetVirtualProjection(self.static_options, Vt) 
        virtual_projection_t_1 = torch_GetVirtualProjection(self.static_options, Vt_1) 

        b, h, w = flo.size()[:3]

        grid_t = torch_GetForwardGrid(self.static_options, real_projection_t, virtual_projection_t)[:,:2,:,:].permute(0,1,3,2)
        grid_t = torch.nn.functional.upsample_bilinear(grid_t, size = (h, w)) # [B,C(xy),H,W]

        grid_t_1 = torch_GetForwardGrid(self.static_options, real_projection_t_1, virtual_projection_t_1)[:,:2,:,:].permute(0,1,3,2) 
        grid_t_1 = torch.nn.functional.upsample_bilinear(grid_t_1, size = (h, w)) # [B,C(xy),H,W]
        
        mesh = self.mesh.repeat(b, 1, 1, 1)
        flo = flo + mesh 
        flo_back = flo_back + mesh # [B,H,W,C]

        valid = (flo[:,:,:,0] > 0) * (flo[:,:,:,1] > 0) * (flo[:,:,:,0] < 1) * (flo[:,:,:,1] < 1)
        valid_f = torch.unsqueeze(valid, dim = 3).type(torch.cuda.FloatTensor)
        valid = torch.unsqueeze(valid, dim = 1).type(torch.cuda.FloatTensor)

        valid_back = (flo_back[:,:,:,0] > 0) * (flo_back[:,:,:,1] > 0) * (flo_back[:,:,:,0] < 1) * (flo_back[:,:,:,1] < 1)
        valid_back_f = torch.unsqueeze(valid_back, dim = 3).type(torch.cuda.FloatTensor) 
        valid_back = torch.unsqueeze(valid_back, dim = 1).type(torch.cuda.FloatTensor) # [B,C,H,W]

        flo = (flo * 2 - 1) * valid_f
        flo_back = (flo_back * 2 - 1) * valid_back_f

        forward_t = torch.nn.functional.grid_sample(grid_t, flo, padding_mode="reflection") # default bilinear
        backward_t_1 = torch.nn.functional.grid_sample(grid_t_1, flo_back, padding_mode="reflection") # default bilinear

        forward_diff = ((forward_t - grid_t_1) * valid) ** 2 
        backward_diff = ((backward_t_1 - grid_t) * valid_back) ** 2

        forward_loss = torch.sum(forward_diff, dim = (1,2,3)) / torch.sum(valid, dim = (1,2,3))
        backward_loss = torch.sum(backward_diff, dim = (1,2,3)) / torch.sum(valid_back, dim = (1,2,3))

        loss = forward_loss + backward_loss
        loss = torch.min(loss, loss - loss + 1) #[0]
        loss = torch.sum(loss) / b

        return loss 


def get_mesh(height = 270, width = 480, USE_CUDA = True):
    xs = np.linspace(0, 1, width, endpoint = False) + 0.5 / height
    ys = np.linspace(0, 1, height, endpoint = False) + 0.5 / width
    xmesh, ymesh = np.meshgrid(xs, ys)
    # Reshape the sampling positions to a H x W x 2 tensor
    mesh = torch.Tensor(np.expand_dims(np.moveaxis(np.array(list(zip(xmesh, ymesh))), 1, 2),axis=0))
    if USE_CUDA:
        mesh = mesh.cuda()
    return mesh

class Undefine_loss(torch.nn.Module):
    def __init__(self, ratio = 0.08, inner_ratio = 0.04, USE_CUDA = True):
        super(Undefine_loss, self).__init__()
        self.static_options = get_static() 
        self.inner_ratio = inner_ratio
        width = self.static_options["width"]
        height = self.static_options["height"]
        x0, x1, y0, y1 = \
            int(width*ratio), int(width*(1-ratio)), int(height*ratio), int(height*(1-ratio))
        self.norm = torch.Tensor([width, height, 1])
        self.p00 = torch.Tensor([x0, y0, 1])
        self.p01 = torch.Tensor([x0, y1, 1])
        self.p10 = torch.Tensor([x1, y0, 1])
        self.p11 = torch.Tensor([x1, y1, 1])
        if USE_CUDA == True:
            self.p00 = self.p00.cuda()
            self.p01 = self.p01.cuda()
            self.p10 = self.p10.cuda()
            self.p11 = self.p11.cuda()
            self.norm = self.norm.cuda()

    def forward(self, Vt, Rt, ratio = 0.04):
        batch_size = Vt.size()[0]

        row_mid = self.static_options["num_grid_rows"] // 2
        virtual_projection_t = torch_GetVirtualProjection(self.static_options, Vt) 

        real_projection_t = torch_GetVirtualProjection(self.static_options, Rt) 

        # virtual projection and real projection
        transform = torch_GetHomographyTransformFromProjections(real_projection_t, virtual_projection_t)
        
        p00 = (torch_ApplyTransform(transform, self.p00) / self.norm)[:,:2]
        p01 = (torch_ApplyTransform(transform, self.p01) / self.norm)[:,:2]
        p10 = (torch_ApplyTransform(transform, self.p10) / self.norm)[:,:2]
        p11 = (torch_ApplyTransform(transform, self.p11) / self.norm)[:,:2]

        loss = torch.stack((self.get_loss(p00), self.get_loss(p01), self.get_loss(p10), self.get_loss(p11)),dim = 1)
        loss,_ = torch.max(loss, dim = 1)

        loss = torch.min(loss, loss - loss + 1) #[0]
        loss = torch.sum(loss) / batch_size

        return loss
    
    def get_loss(self, p):
        d =  (p - self.inner_ratio) * (p < self.inner_ratio).type(torch.cuda.FloatTensor) + \
            (1 - self.inner_ratio - p) * (p > (1 - self.inner_ratio)).type(torch.cuda.FloatTensor)
        return torch.sum(d**2, dim = 1) 
