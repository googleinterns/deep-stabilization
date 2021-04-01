import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import torch
import cv2
import time

device = torch.device("cuda")

def Rasterization(image, grid, get_mesh_only = False):
    # grid xy WH
    shape = image.size()
    height = shape[1]
    width = shape[2]
    wapper_upper_triangle, wapper_lower_triangle = grid_to_triangle(grid[:,:,:2])
    origin_upper_triangle, origin_lower_triangle = grid_to_triangle(grid[:,:,2:])


    [xmax, xmin, ymax, ymin], xlength, ylength = grid_size(wapper_upper_triangle, wapper_lower_triangle, height, width)

    xratio = xlength / width
    yratio = ylength / height

    wapper_triangle = torch.stack((wapper_upper_triangle,wapper_lower_triangle),dim = 1).to(device) # grid * upper/lower * point * xy
    origin_triangle = torch.stack((origin_upper_triangle,origin_lower_triangle),dim = 1).to(device) # grid * upper/lower * point * xy

    tran_triangle = torch.zeros(wapper_triangle.size()).to(device)

    tran_triangle[:,:,:,0] = (wapper_triangle[:,:,:,0] - xmin.view(-1,1,1).to(device)/width) / xratio
    tran_triangle[:,:,:,1] = (wapper_triangle[:,:,:,1] - ymin.view(-1,1,1).to(device)/height) / yratio

    mask = triangle2mask(tran_triangle, ylength, xlength) # consuming

    mask = torch.unsqueeze(mask, 4)
    origin_triangle = torch.unsqueeze(origin_triangle, 1)

    grid_sample = origin_triangle * mask # consuming
    grid_sample = torch.sum(torch.sum(grid_sample, dim = 3), dim = 2).view(-1,ylength,xlength,2) # consuming

    gxmin = min(0, int(torch.min(xmin)))
    gxmax = int(torch.max(xmin) + xlength)
    gymin = min(0, int(torch.min(ymin)))
    gymax = int(torch.max(ymin) + ylength)
    grid_merge = torch.zeros((max(gymax-gymin, height, height - gymin),max(gxmax - gxmin, width, width - gxmin),2)).to(device)
    for i in range(grid_sample.size()[0]):
        x_s = int(xmin[i] - gxmin)
        x_e = int(xmin[i] + xlength - gxmin)
        y_s = int(ymin[i] - gymin)
        y_e = int(ymin[i] + ylength -gymin)
        grid_merge[ y_s:y_e, x_s:x_e, :] += grid_sample[i, :, :, :]

    # grid_merge = grid_merge[min(-gxmin,0):min(-gxmin,0)+height, min(-gymin,0):min(-gymin,0)+width, :] 
    grid_merge = grid_merge[-gymin:-gymin+height, -gxmin:-gxmin+width, :] 
    # if get_mesh_only:
    #     grid_merge = grid_merge.cpu().numpy()
    #     mesh_grid = generate_mesh_grid(height, width)
    #     out = grid_merge - mesh_grid
    #     return np.concatenate((out[:,:,1:],out[:,:,:1]),2)
    
    shift = torch.tensor([0.5/height,0.5/width])[None, None, :].to(device)
    grid_merge = (grid_merge + 1*shift) * 2 - 1

    image[:3,:2,:2] = 0

    image = torch.unsqueeze(image, 0).to(device)
    grid_merge = torch.unsqueeze(grid_merge, 0)

    image = torch.nn.functional.grid_sample(image, grid_merge) # default bilinear

    image = torch.squeeze(image, 0)
    return image.cpu()

def grid_to_triangle(grid):
    grid_shape = grid.size()
    num = (grid_shape[0] - 1) * (grid_shape[1] - 1)

    upper_triangle = grid[:-1, :-1, :, None]
    upper_triangle = torch.cat(( upper_triangle, grid[1:, :-1, :, None]), dim = 3)
    upper_triangle = torch.cat(( upper_triangle, grid[:-1, 1:, :, None]), dim = 3)
    upper_triangle = upper_triangle.view(num, 2, 3)
    upper_triangle = torch.transpose(upper_triangle, 1, 2) # grid * point * xy
 
    lower_triangle = grid[:-1, 1:, :, None]
    lower_triangle = torch.cat(( lower_triangle, grid[1:, :-1, :, None]), dim = 3)
    lower_triangle = torch.cat(( lower_triangle, grid[1:, 1:, :, None]), dim = 3)
    lower_triangle = lower_triangle.view(num, 2, 3)
    lower_triangle = torch.transpose(lower_triangle, 1, 2)
    
    return upper_triangle,  lower_triangle # grid * point * xy

def grid_size(upper_triangle, lower_triangle, height, width):
    wapper_grid = torch.cat((upper_triangle, lower_triangle),dim =1)
    xmax = torch.floor(torch.max(wapper_grid[:,:,0]*width, 1)[0]) + 1
    ymax = torch.floor(torch.max(wapper_grid[:,:,1]*height, 1)[0]) + 1
    xmin = torch.floor(torch.min(wapper_grid[:,:,0]*width, 1)[0])
    ymin = torch.floor(torch.min(wapper_grid[:,:,1]*height, 1)[0])

    xlength = int(torch.max(xmax - xmin))
    ylength = int(torch.max(ymax - ymin))

    return [xmax, xmin, ymax, ymin], xlength, ylength

def generate_mesh_grid(height, width):
    # Create a grid of sampling positions
    xs = np.linspace(0, 1, width, endpoint=False)
    ys = np.linspace(0, 1, height, endpoint=False)
    xmesh, ymesh = np.meshgrid(xs, ys)
    # Reshape the sampling positions to a H x W x 2 tensor
    return np.moveaxis(array(list(zip(xmesh, ymesh))), 1, 2)

def triangle2mask(d, height, width): # d: [N x T x 3 x 2]
    N = d.size()[0] # batch size
    T = d.size()[1] # triangle number
    P = height * width # The number of pixels in the output image.

    area = edgefunc(d[:, :, 1, :], d[:, :, 2, :], d[:, :, None, 0, :])

    gridcpu = generate_mesh_grid(height, width)
    
    gridcpu = np.reshape(gridcpu, (height*width, 2))

    grid = torch.Tensor(gridcpu)
    grid = grid.unsqueeze(0).repeat((N, T, 1, 1)) # [N x T x P x 2]

    grid = grid.to(device)

    # Evaluate the edge functions at every position.
    # We should get a [N x P] vector out of each.
    w0 = edgefunc(d[:, :, 1, :], d[:, :, 2, :], grid) / area
    w1 = edgefunc(d[:, :, 2, :], d[:, :, 0, :], grid) / area
    w2 = edgefunc(d[:, :, 0, :], d[:, :, 1, :], grid) / area

    # Only pixels inside the triangles will have color
    # [N x P]

    mask = (w0 > 0) & (w1 > 0) & (w2 > 0)
    mask = torch.unsqueeze(mask, 3).type(torch.cuda.FloatTensor)

    w = torch.stack((w0,w1,w2),dim = 3) * mask

    return torch.transpose(w, 1, 2) # [N x P x T x 3]
    

def edgefunc(v0, v1, p):
    """
    let P = H * W
    v0 and v1 have vertex positions for all T triangles.
    Their shapes are [N x T X 2]
    p is a list of sampling points as a [N x T X P x 2] tensor.
    Each of the T triangles has an [P x 2] matrix of sampling points.
    returns a [N x T x P] matrix
    """
    P = p.size()[2]
    
    # Take all the x and y coordinates of all the positions as a
    # [N x S] tensor
    py = p[:, :, :, 1]
    px = p[:, :, :, 0]

    # We need to manually broadcast the vector to cover all sample points
    x10 = v0[:, :, 0] - v1[:, :, 0] # [N x T]
    y01 = v1[:, :, 1] - v0[:, :, 1] # [N x T]

    x10 = x10.unsqueeze(2).repeat((1, 1, P)) # [N x T x P]
    y01 = y01.unsqueeze(2).repeat((1, 1, P)) # [N x T x P]

    cross = v0[:,:,1]*v1[:,:,0] - v0[:,:,0]*v1[:,:,1] # [N x T]
    cross = cross.unsqueeze(2).repeat((1, 1, P)) # [N x T x P]

    return y01*px + x10*py + cross

if __name__ == '__main__':
    print(generate_mesh_grid(2,3))