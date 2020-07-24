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
from read_write import visialize
        
        
class C2_Smooth_loss(torch.nn.Module):
    def __init__(self):
        super(C2_Smooth_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, Qt, Qt_1, Qt_2):
        # Mt = torch_ConvertQuaternionToRotationMatrix(Qt)
        # Mt_1 = torch_ConvertQuaternionToRotationMatrix(Qt_1)
        # Mt_2 = torch_ConvertQuaternionToRotationMatrix(Qt_2)
        # detaQt = torch_ConvertRotationMatrixToQuaternion(torch.matmul(Mt, torch.inverse(Mt_1)))
        # detaQt_1 = torch_ConvertRotationMatrixToQuaternion(torch.matmul(Mt_1, torch.inverse(Mt_2)))
        # Rt = torch_ConvertAxisAngleToQuaternion(Qt) 
        # Rt_1 = torch_ConvertAxisAngleToQuaternion(Qt_1)
        # Rt_2 = torch_ConvertAxisAngleToQuaternion(Qt_2)
        detaQt = torch_QuaternionProduct(Qt, torch_QuaternionReciprocal(Qt_1))  
        detaQt_1 = torch_QuaternionProduct(Qt_1, torch_QuaternionReciprocal(Qt_2))  
        return self.MSE(detaQt, detaQt_1)

class C1_Smooth_loss(torch.nn.Module):
    def __init__(self):
        super(C1_Smooth_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, v_r_axis, v_axis_t_1, real_postion = None):
        # v_r_Q = torch_ConvertAxisAngleToQuaternion(v_r_axis)
        # v_Q = torch_QuaternionProduct(real_postion, v_r_axis)
        # v_Q_t_1 = torch_ConvertAxisAngleToQuaternion(v_axis_t_1)
        # return self.MSE(v_Q, v_Q_t_1)
        return self.MSE(v_r_axis, v_axis_t_1)

class Angle_loss(torch.nn.Module):
    def __init__(self):
        super(Angle_loss, self).__init__()
        self.MSE = torch.nn.MSELoss()

    def forward(self, Q1, Q2, threshold = 0.5236, logistic_beta1 = 100):
        Q3 = torch_norm_quat(torch_QuaternionProduct(Q2, torch_QuaternionReciprocal(Q1)))
        theta = torch.acos(Q3[:,3]) * 2
        loss = torch.mean(theta * (1 / (1 + torch.exp(-logistic_beta1 * (theta - threshold)))))

        # batch_size = Q3.size()[0]
        # loss = Variable(torch.zeros((1), requires_grad=True))
        # for i in range(batch_size):
        #     if angle[i] > threshold:
        #         loss += 1
        return loss, theta

class Optical_loss(torch.nn.Module):
    def __init__(self, loss_optical_w = 1, loss_undefine_w = 1):
        super(Optical_loss, self).__init__()
        self.loss_optical_w = loss_optical_w
        self.loss_undefine_w = loss_undefine_w

    def forward(self, Vt, Vt_1, flo, flo_back, real_projection_t, real_projection_t_1):
        static_options = get_static() # TODO: May need to change

        virtual_projection_t = torch_GetVirtualProjection(static_options, Vt) 
        virtual_projection_t_1 = torch_GetVirtualProjection(static_options, Vt_1) 

        b, h, w = flo.size()[:3]

        grid_t = torch_GetForwardGrid(static_options, real_projection_t, virtual_projection_t)[:,:2,:,:].permute(0,1,3,2)
        grid_t = torch.nn.functional.upsample_bilinear(grid_t, size = (h, w)) # [B,C(xy),H,W]

        grid_t_1 = torch_GetForwardGrid(static_options, real_projection_t_1, virtual_projection_t_1)[:,:2,:,:].permute(0,1,3,2) 
        grid_t_1 = torch.nn.functional.upsample_bilinear(grid_t_1, size = (h, w)) # [B,C(xy),H,W]

        # real = torch_GetWarpingFlow(static_options, real_projection_t_1, real_projection_t)[:,:2,:,:]
        # real = torch.nn.functional.upsample_bilinear(real, size = (h, w))

        mesh = get_mesh(b, h, w)

        flo = flo + mesh
        flo_back = flo_back + mesh # [B,H,W,C]

        print(flo)
        self.visualize_figure("flo.png", mesh, flo)
        assert(False)
        valid = (flo[:,:,:,0] > 0) * (flo[:,:,:,1] > 0) * (flo[:,:,:,0] < 1) * (flo[:,:,:,1] < 1)
        valid = torch.unsqueeze(valid, dim = 1).type(torch.cuda.FloatTensor)

        valid_back = (flo_back[:,:,:,0] > 0) * (flo_back[:,:,:,1] > 0) * (flo_back[:,:,:,0] < 1) * (flo_back[:,:,:,1] < 1)
        valid_back = torch.unsqueeze(valid_back, dim = 1).type(torch.cuda.FloatTensor) # [B,C,H,W]

        flo = flo * 2 - 1
        flo_back = flo_back * 2 - 1

        forward_t = torch.nn.functional.grid_sample(grid_t, flo) # default bilinear
        backward_t_1 = torch.nn.functional.grid_sample(grid_t_1, flo_back) # default bilinear


        forward_diff = ((forward_t - grid_t_1) * valid) ** 2 
        backward_diff = ((backward_t_1 - grid_t) * valid_back) ** 2

        forward_loss = torch.sum(forward_diff, dim = (1,2,3)) / torch.sum(valid, dim = (1,2,3))
        backward_loss = torch.sum(backward_diff, dim = (1,2,3)) / torch.sum(valid_back, dim = (1,2,3))
        
        return torch.sum(forward_loss + backward_loss)

    def visualize_point(self, forward_t, grid_t_1, backward_t_1, grid_t):
        forward_t = self.sample_data(forward_t)
        grid_t_1 = self.sample_data(grid_t_1)
        backward_t_1 = self.sample_data(backward_t_1)
        grid_t = self.sample_data(grid_t)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(forward_t[:,0],forward_t[:,1],'r.')
        plt.plot(grid_t_1[:,0],grid_t_1[:,1],'g.')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('forward flow frame ')
        plt.subplot(122)
        plt.plot(backward_t_1[:,0],backward_t_1[:,1],'r.')
        plt.plot(grid_t[:,0],grid_t[:,1],'g.')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('backward flow frame ')
        plt.savefig("./no_virtual_warp_follow.jpg")

    def visualize_figure(self, save_path, grid1, grid2):
        grid1 = self.sample_data(grid1)
        grid2 = self.sample_data(grid2, second = True)
        p1 = "/mnt/disks/dataset/Google/train/s2_outdoor_runing_forward_VID_20200304_144434/frames/frame_0000.png"
        p2 = "/mnt/disks/dataset/Google/train/s2_outdoor_runing_forward_VID_20200304_144434/frames/frame_0001.png"
        visialize(save_path,p1,p2, grid1, grid2)

    def sample_data(self, flow, h = 1080, w = 1920 ,second = False):
        flow = flow[0].cpu().numpy()[27:250:27,48:440:48,:]
        flow = np.reshape(flow, (-1, 2))
        flow[:,0] = flow[:,0] * w 
        flow[:,1] = flow[:,0] * h
        if second:
            flow[:,1] += h 
        print(flow)
        flow.astype(np.int)
        return flow

def get_mesh(batch, height, width, USE_CUDA = True):
    xs = np.linspace(0, 1, width, endpoint = False) + 0.5 / height
    ys = np.linspace(0, 1, height, endpoint = False) + 0.5 / width
    xmesh, ymesh = np.meshgrid(xs, ys)
    # Reshape the sampling positions to a H x W x 2 tensor
    mesh = torch.Tensor(np.expand_dims(np.moveaxis(np.array(list(zip(xmesh, ymesh))), 1, 2),axis=0))
    if USE_CUDA:
        mesh = mesh.cuda()
    return mesh.repeat(batch, 1, 1, 1)

class Undefine_loss(torch.nn.Module):
    def __init__(self):
        super(Undefine_loss, self).__init__()

    def forward(self, Vt, real_projection_t, h = 1080, w = 1920, USE_CUDA = True):
        static_options = get_static() 
        width = static_options["width"]
        height = static_options["height"]

        row_mid = static_options["num_grid_rows"] // 2
        virtual_projection_t = torch_GetVirtualProjection(static_options, Vt) 
        

        # grid_t = torch_GetForwardGrid(static_options, real_projection_t, virtual_projection_t)[:,:2,:,:].permute(0,1,3,2)
        # grid_t = torch.nn.functional.upsample_bilinear(grid_t, size = (h, w)).permute(0,2,3,1) # [B,H,W,C]

        # virtual projection and real projection
        transform = torch_GetHomographyTransformFromProjections(virtual_projection_t, real_projection_t[:, row_mid])
        x0, x1, y0, y1 = int(width*0.1), int(width*0.9), int(height*0.1), int(height*0.9)
        
        norm = torch.Tensor([width, height, 1])
        p00 = torch.Tensor([x0, y0, 1])
        p01 = torch.Tensor([x0, y1, 1])
        p10 = torch.Tensor([x1, y0, 1])
        p11 = torch.Tensor([x1, y1, 1])
        if USE_CUDA == True:
            p00 = p00.cuda()
            p01 = p01.cuda()
            p10 = p10.cuda()
            p11 = p11.cuda()
            norm = norm.cuda()
        p00 = (torch_ApplyTransform(transform, p00) / norm)[:,:2]
        p01 = (torch_ApplyTransform(transform, p01) / norm)[:,:2]
        p10 = (torch_ApplyTransform(transform, p10) / norm)[:,:2]
        p11 = (torch_ApplyTransform(transform, p11) / norm)[:,:2]

        loss = torch.zeros((1)).cuda()
        for i in range(Vt.size()[0]):
            loss += torch.max(torch.stack((self.get_loss(p00[i]), self.get_loss(p01[i]), self.get_loss(p10[i]), self.get_loss(p11[i])),dim = 0))
        return loss

    def get_loss(self, p):
        if p[0] < 0: d0 = p[0]
        elif p[0] > 1: d0 = 1 - p[0]
        else: d0 = p[0] * 0
    
        if p[1] < 0: d1 = p[1]
        elif p[1] > 1: d1 = 1 - p[1]
        else: d1 = p[1] * 0
        return torch.sum(torch.abs(torch.stack((d0,d1), dim = 0)))

    def compare_0(self, tensor): 
        return torch.abs(tensor) - tensor

    def compare_1(self, tensor):
        tensor = 1 - tensor
        return torch.abs(tensor) - tensor


