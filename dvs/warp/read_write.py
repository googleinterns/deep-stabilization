import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import ffmpeg  
import json
import torch 
import argparse
    
def load_video(path, save_dir = None, resize = None, length = -1): # N x H x W x C
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    print(image.shape)
    height, width, layers = image.shape
    if resize is None:
        size = (width,height)
    elif type(resize) is int:
        size = (width//resize,height//resize)
    else:
        size = resize
    count = 0
    frames = []
    while success:  
        if resize is not None:
            image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR)
        if save_dir != None:
            path = os.path.join(save_dir, "frame_" + str(count).zfill(4) + ".png")
            cv2.imwrite(path, image) 
        frames.append(image)
        success,image = vidcap.read()
        count += 1
        if length > 0 and count >= length:
            break
    print("Video length: ", len(frames))
    return frames, fps, size

def video2frame(path, resize = None):
    data_name = sorted(os.listdir(path))
    for i in range(len(data_name)):
        print(str(i+1)+" / " + str(len(data_name)))
        data_folder = os.path.join(path, data_name[i])
        print(data_folder)
        files = os.listdir(data_folder)
        for f in files:
            if f[-4:] == ".mp4":
                video_name = f
        video_path = os.path.join(data_folder, video_name)
        frame_folder = os.path.join(data_folder, "frames")
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)
        load_video(video_path, save_dir = frame_folder, resize=resize)

def video2frame_one_seq(path, save_dir = None, resize = None): # N x H x W x C
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    print(path)
    print(image.shape)	
    height, width, layers = image.shape
    if resize is None:
        size = (width,height)
    elif type(resize) is int:
        size = (width//resize,height//resize)
    else:
        size = resize
    count = 0
    while success:  
        if resize is not None:
            image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR)
        if save_dir != None:
            path = os.path.join(save_dir, "frame_" + str(count).zfill(5) + ".png")
            cv2.imwrite(path, image) 
        success,image = vidcap.read()
        count += 1
    return fps, size

def save_video(path,frame_array, fps, size, losses = None, frame_number = False, writer = None):
    if writer is None:
        if path[-3:] == "mp4":
            out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        else:
            out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    else:
        out = writer
    for i in range(len(frame_array)):
        # writing to a image array
        if frame_number:
            frame_array[i] = draw_number(np.asarray(frame_array[i]), i)
        if losses is not None:
            frame_array[i] = draw_number(np.asarray(frame_array[i]), losses[i], x = 900, message = "Loss: ")
        out.write(frame_array[i])
    if writer is None:
        out.release()

def draw_number(frame, num, x = 10, y = 10, message = "Frame: "):
    image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./data/arial.ttf", 45)
     
    message = message + str(num)
    color = 'rgb(0, 0, 0)' # black color
    
    draw.text((x, y), message, fill=color, font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("FlowNet2 Preparation")
    parser.add_argument("--dir_path", default="./video")
    args = parser.parse_args()
    dir_path = args.dir_path
    if dir_path == "./video":
        video2frame(dir_path, resize = 4)
    else:
        video2frame(os.path.join(dir_path, "test"), resize = 4)
        video2frame(os.path.join(dir_path, "training"), resize = 4)