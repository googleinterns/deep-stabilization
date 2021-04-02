import numpy as np
from .read_write import load_video, save_video
import torch
import cv2
from .rasterizer import Rasterization
import time
import os

def warp_video(mesh_path, video_path, save_path, losses = None, frame_number = False, fps_fix = None):
    if type(mesh_path) == str:
        print("Error")
    else:
        grid_data = mesh_path

    frame_array, fps, size = load_video(video_path, length = grid_data.shape[0])
    if fps_fix is not None:
        fps = fps_fix
    length = min(grid_data.shape[0], len(frame_array))
    seq_length = 100
    seq = length//seq_length
    writer = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(seq+1):
        if seq_length*i==length:
            break
        print("Frame: "+str(i*seq_length)+"/"+str(length))
        frame_array_save = warpping_rast(grid_data[seq_length*i:min(seq_length*(i+1),length)], frame_array[seq_length*i:min(seq_length*(i+1),length)], losses = losses)
        save_video(save_path,frame_array_save, fps, size, losses = losses, frame_number = frame_number, writer = writer)
    writer.release()

def warpping_rast(grid_data, frame_array, losses = None):
    output = []
    for i in range(0, min(len(frame_array), grid_data.shape[0])):
        frame = warpping_one_frame_rast(frame_array[i], grid_data[i])
        output.append(frame)
    return output

def warpping_one_frame_rast(image, grid):
    img = torch.Tensor(image).permute(2,0,1)/255
    grid = torch.Tensor(grid)
    output_image = Rasterization(img, grid)
    return np.clip(output_image.permute(1,2,0).numpy() * 255, 0, 255).astype("uint8")
