import numpy as np
from .read_write import load_mesh, load_video, save_video
import torch
import torchgeometry as tgm
import cv2
from .rasterizer import Rasterization
import time
from .render import rendering
from flownet2 import flow_utils
import os


# def warp_video_opt(mesh_path, flo, video_path, save_path):
#     if type(mesh_path) == str:
#         mesh_data = load_gyro_mesh(mesh_path)
#         grid_data = mesh_data["warping grid"]
#         # mesh_data = load_mesh(mesh_path)
#     else:
#         grid_data = mesh_path

#     # Need to change order of xy to be yx
#     grid_data = np.stack((grid_data[:,:,:,1], grid_data[:,:,:,0],grid_data[:,:,:,3],grid_data[:,:,:,2]), axis =3)
    
#     frame_array, fps, size = load_video(video_path)
#     length = min(grid_data.shape[0], len(frame_array))
#     t1 = time.time()
#     # frame_array = rendering(mesh_data[:length], frame_array[:length], size)
#     frame_array = warpping_rast_opt(grid_data[:length], frame_array[:length], flo[:length])
#     t2 = time.time()
#     print(t2-t1)
#     save_video(save_path,frame_array, fps, size)

# def warpping_one_frame_rast_opt(image, grid, flo):
#     img = torch.Tensor(image).permute(2,0,1)/255
#     grid = torch.Tensor(grid)
#     t1 = time.time()
#     flo = flow_utils.readFlow(flo)
#     output_image = Rasterization(img, grid,  get_mesh_only = True)
#     t2 = time.time()
#     print(t2 - t1)
#     return flo*output_image

# def warpping_rast_opt(grid_data, frame_array, flo):
#     output = []
#     t1 = time.time()
#     path = "./test/opt"
#     for i in range(1, min(len(frame_array), grid_data.shape[0])):
#         if i % 20 == 0:
#             print(i)
#         frame = warpping_one_frame_rast_opt(frame_array[i], grid_data[i], flo[i-1])
#         flow_utils.writeFlow( os.path.join(path, '%06d.flo' % (i)),  frame)
#         flow_utils.visulize_flow_file(os.path.join(path, '%06d.flo' % (i)),"./test/opt_vis")
#         output.append(frame)
#     return output

def warp_video(mesh_path, video_path, save_path, losses = None, frame_number = False, fps_fix = None):
    if type(mesh_path) == str:
        print("Error")
        # mesh_data = load_gyro_mesh(mesh_path)
        # grid_data = mesh_data["warping grid"]
        # mesh_data = load_mesh(mesh_path)
    else:
        grid_data = mesh_path

    frame_array, fps, size = load_video(video_path)
    if fps_fix is not None:
        fps = fps_fix
    length = min(grid_data.shape[0], len(frame_array))
    # frame_array = rendering(mesh_data[:length], frame_array[:length], size)
    seq_length = 100
    seq = length//seq_length
    writer = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(seq+1):
        if seq_length*i==length:
            break
        print("Seq: ",i)
        frame_array_save = warpping_rast(grid_data[seq_length*i:min(seq_length*(i+1),length)], frame_array[seq_length*i:min(seq_length*(i+1),length)], losses = losses)
        save_video(save_path,frame_array_save, fps, size, losses = losses, frame_number = frame_number, writer = writer)
    writer.release()

def warpping_rast(grid_data, frame_array, losses = None):
    output = []
    for i in range(0, min(len(frame_array), grid_data.shape[0])):
        if i % 20 == 0:
            print(i)
        t1 = time.time()
        frame = warpping_one_frame_rast(frame_array[i], grid_data[i])
        t2 = time.time()
        print(t2 - t1)
        output.append(frame)
    return output

def warpping_one_frame_rast(image, grid):
    img = torch.Tensor(image).permute(2,0,1)/255
    grid = torch.Tensor(grid)
    output_image = Rasterization(img, grid)
    return np.clip(output_image.permute(1,2,0).numpy() * 255, 0, 255).astype("uint8")


def warpping_homo(mesh_data, frame_array, size):
    width, height = size[0], size[1]
    rows = mesh_data[0]["vertex_grid_rows"]
    warper = tgm.HomographyWarper(height//rows, width)
    output = []
    for i in range(len(frame_array)):
        output.append(warpping_one_frame_homo(warper, frame_array[i], torch.Tensor(mesh_data[i]["per-row homography"]), rows))
    return output

def warpping_one_frame_homo(warper, image, homo, rows):
    img = torch.Tensor(image).permute(2,0,1)
    shape = img.size()
    height = shape[1]
    img = torch.unsqueeze(img.float(), dim=0)  # BxCxHxW
    output_image = torch.zeros(shape[0], shape[1],shape[2])
    for i in range(rows):
        # output_image[:,height//rows*i:height//rows*(i+1),:] = warper(img[:,:,height//rows*i:height//rows*(i+1),:], torch.tensor(np.linalg.inv(homo[i:i+1])))
        output_image[:,height//rows*i:height//rows*(i+1),:] = warper(img[:,:,height//rows*i:height//rows*(i+1),:], homo[i:i+1])
    return output_image.permute(1,2,0).numpy().astype("uint8")

def resize(img):
    print(img.shape)
    img = img / 255
    shift = 0
    x = 108
    y = 192
    # img = cv2.resize(img, (1920*2,1080*2), interpolation = cv2.INTER_LINEAR)
    img = img[1*x+shift:1*(1080-x)+shift, 1*y+shift:1*(1920-y)+shift, :]
    img = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_LINEAR)
    print(img.shape)
    return img*255

def normalize(img):
    img = img*255/np.max(img)
    return img


if __name__ == "__main__":
    def test_rast():
        img = cv2.imread("./result/frame100_r.jpg").astype(np.float64)
        img2 = cv2.imread("./result/frame100_i.jpg").astype(np.float64)
        mesh_data = load_mesh("./data/testdata/results_full_identity/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt")
        grid_data = mesh_data["warping grid"]
        grid_data = np.stack((grid_data[:,:,:,1], grid_data[:,:,:,0],grid_data[:,:,:,3],grid_data[:,:,:,2]), axis =3)
        grid = grid_data[100]
        img = warpping_one_frame_rast(img, grid).astype(int)
        # img = resize(img)
        diff = np.abs(img - img2).astype("uint8")
        print(np.sum(diff.astype("uint8")))
        print(np.max(diff))
        diff = normalize(diff).astype("uint8")
    
        cv2.imwrite("diff.jpg", diff) 
        cv2.imwrite("out.jpg", img) 
    test_rast()
    # gyro_path = './data/testdata/videos_with_zero_virtual_motion/results_no_rotation/s2_outdoor_runing_forward_VID_20200304_144227_stab_mesh.txt'
    # mesh_path = "./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt"
    # video_path = "./data/testdata//videos_with_zero_virtual_motion/inputs_no_rotation/s2_outdoor_runing_forward_VID_20200304_144227/s2_outdoor_runing_forward_VID_20200304_144227.mp4"
    # warp_video(gyro_path, video_path, "./no_rotat.mp4")
