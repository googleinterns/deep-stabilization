import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import ffmpeg  
import json

def load_mesh(path):
    f = open(path, "r")
    read_data = f.read().split("\n")
    f.close()
    mesh_data = {}
    # count = 0 
    while len(read_data) > 0 and len(read_data[0]) > 0:
        assert(read_data[0][:len("frame id")] == "frame id")
        read_data.pop(0)
        dic = load_frame_in_mesh(read_data)
        for key in dic:
            if key not in mesh_data:
                mesh_data[key] = np.expand_dims(np.array(dic[key]), axis=0)
            else:
                mesh_data[key] = np.concatenate((mesh_data[key], np.expand_dims(np.array(dic[key]), axis=0)), axis=0)
        # count += 1 
        # if count > 4:
        #     break
    print("Mesh length: ", len(list(mesh_data.values())[0]))
    return mesh_data


def load_frame_in_mesh(read_data):
    dic = {}
    grid_num = read_data.pop(0).split(", ")
    dic["vertex_grid_rows"] = int(grid_num[0][len("vertex_grid_rows")+1:])
    dic["vertex_grid_cols"] = int(grid_num[1][len("vertex_grid_cols")+1:])

    read_data.pop(0) # warping grid
    warp_grid = read_data.pop(0).split(" ")
    if warp_grid[-1] == "":
        warp_grid.pop(-1)

    dic["warping grid"] = np.array([float(i) for i in warp_grid]).reshape((dic["vertex_grid_rows"],dic["vertex_grid_cols"],4))

    read_data.pop(0) # per-row homography
    dic["per-row homography"] = []
    for i in range(dic["vertex_grid_rows"]):
        warp_row = read_data.pop(0).split(" ")
        if warp_row[-1] == "":
            warp_row.pop(-1)
        dic["per-row homography"] += [float(i) for i in warp_row]
    dic["per-row homography"] = np.array(dic["per-row homography"]).reshape((dic["vertex_grid_rows"],3,3))
    return dic

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 
    
def load_video(path, save_dir = None, resize = None): # N x H x W x C
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    print(image.shape)
    height, width, layers = image.shape
    if resize is None:
        size = (width,height)
    else:
        size = (width//resize,height//resize)
    count = 0
    frames = []
    while success:  
        # if count> 140:
        #     break
        if save_dir != None:
            path = os.path.join(save_dir, "frame_" + str(count).zfill(4) + ".png")
            if resize is not None:
                image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(path, image) 
        frames.append(image)
        success,image = vidcap.read()
        count += 1
        # if count > 4:
        #     break
    print("Video length: ", len(frames))
    return frames, fps, size

def video2frame(path):
    data_name = sorted(os.listdir(path))
    for i in range(len(data_name)):
        print(str(i+1)+" / " + str(len(data_name)))
        data_folder = os.path.join(path, data_name[i])
        print(data_folder)
        files = os.listdir(data_folder)
        for f in files:
            if f[-7:] == "ois.mp4":
                video_name = f
        video_path = os.path.join(data_folder, video_name)
        frame_folder = os.path.join(data_folder, "frames")
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)
        load_video(video_path, save_dir = frame_folder, resize=4)

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

def compare_difference(path1, path2):
    frames1, fps, size = load_video(path1)
    frames2, _, _ = load_video(path2)
    length = min(len(frames1),len(frames2),60)

    sum_total = 0
    diff_total = 0

    out = cv2.VideoWriter("./result/diff.mp4",cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(length):
        # writing to a image array
        if i % 20 == 0:
            print(i)
        frame, diff_batch, sum_batch = get_diff(frames1[i], frames2[i])
        sum_total += sum_batch
        diff_total += diff_batch
        frame = frame.astype("uint8")
        frame = draw_number(np.asarray(frame), i)
        out.write(frame)
    out.release()

    print(sum_total)
    print(diff_total)


def get_diff(frame1, frame2):
    frames1 = np.array(frame1).astype(int)
    frames2 = np.array(frame2).astype(int)
    diff = np.abs(frames1 - frames2)
    return diff, np.sum(diff), np.sum(np.abs(frames1) + np.abs(frames2))

def visialize(save_path, image1_path, image2_path, grid1, grid2):
    img = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    h, w, _ = img.shape
    image = np.concatenate((img, img2), axis = 0)

    image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    
    for i in range(len(grid1)):
        p1 = (grid1[i,0], grid1[i,1])
        p2 = (grid2[i,0], grid2[i,1])
        draw.line(xy = (p1,p2), fill = (255,0,0), width = 1)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image) 

def visualize_flow(save_path, p1, p2, grid1, grid2):
    img = cv2.imread(p1)
    h, w, _ = img.shape

    grid1 = sample_data(grid1, h = h, w = w)
    grid2 = sample_data(grid2, h = h, w = w, second = True)
    visialize(save_path, p1, p2, grid1, grid2)

def sample_data(flow, h = 270, w = 480 ,second = False):
    dense = 8
    flow_h, flow_w, _ = flow.shape
    grid = np.zeros((dense,dense,2))
    h_gap = flow_h//dense
    w_gap = flow_w//dense
    for i in range(dense):
        sub = flow[h_gap * (i+1) -1, 6*i:-1:w_gap,:]
        grid[i,:,:] = sub
    grid = np.reshape(grid, (-1, 2))
    grid[:,0] = grid[:,0] * w 
    grid[:,1] = grid[:,1] * h
    if second:
        grid[:,1] += h 
    grid = grid.astype(np.int)
    return grid

def visualize_point(forward_t, grid_t_1, backward_t_1, grid_t):
    forward_t = sample_data(forward_t)
    grid_t_1 = sample_data(grid_t_1)
    backward_t_1 = sample_data(backward_t_1)
    grid_t = sample_data(grid_t)
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

# def load_EIS_result(video_name):
#     path = "/mnt/disks/dataset/EIS2020_new/results/"+video_name+"_stab.txt"
#     gyro = []
#     with open(path) as json_file:
#         data = json.load(json_file)
#     print(data)
#     for i in range(len(data["frames"])):
#         # print(data["frames"][i]["gyro_nonlinear"].keys)
#         gyro.append(data["frames"][i]["gyro_nonlinear"]["real_cam_pose"])
#         # gyro.append(data["frames"][i]["gyro_nonlinear"]["virtual_cam_pose"])
#     return gyro

if __name__ == "__main__":
    pass
    # mesh_data = load_mesh("./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt")
    # frame_array, fps, size = load_video("./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_feature.mp4")
    # save_video("./out.mp4",frame_array, fps, size)
    # p1 = "./result/output3.mp4"
    # p2 = "./test/sample_test/s2_outdoor_runing_forward_VID_20200304_144227_stab.mp4"
    # p3 = "./data/testdata/videos_with_zero_virtual_motion/results_no_rotation/s2_outdoor_runing_forward_VID_20200304_144227_stab.mp4"
    # compare_difference(p3,p2)
    # p4 = "/mnt/disks/dataset/Google_ois/test/s0_outdoor_panning_quick_VID_20200304_143946/s0_outdoor_panning_quick_VID_20200304_143946.mp4"
    # p5 = "./s1_walking.mp4"
    # frames, fps, size = load_video(p5)
    # print(np.array(frames).shape)
    # save_video("s1_walking_short.mp4",frames,fps,size)
    # result_poses_name = './data/testdata/results_updated/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt'
    # result_poses = LoadStabResult(result_poses_name)
    # grid1 = mesh_data["warping grid"]
    # grid2 = np.reshape(result_poses['warping grid'],(-1,12,12,4))
    # print(np.sum(np.abs(grid1 - grid2)))
    # print(np.sum(np.abs(grid1) + np.abs(grid2)))
    video2frame("./video")
    # p1 = "/mnt/disks/dataset/Google_ois/test/s2_outdoor_runing_forward_VID_20200304_144434/s2_outdoor_runing_forward_VID_20200304_144434_no_ois.mp4"
    # p2 = "/mnt/disks/dataset/Google_ois/test/s2_outdoor_runing_forward_VID_20200304_144434/s2_outdoor_runing_forward_VID_20200304_144434_check.mp4"
    # p1 = "./test/dr_vir_undefine_bk/s2_outdoor_runing_forward_VID_20200304_144434_stab.mp4"
    # p2 = "./test/dr_vir_undefine/s2_outdoor_runing_forward_VID_20200304_144434_stab.mp4"
    # compare_difference(p1,p2)
    # flow_path = "/mnt/disks/dataset/Google_ois/test/s0_outdoor_panning_quick_VID_20200304_143946/flo/000220.flo"
    # f = flow_utils.readFlow(flow_path).astype(np.float32) 
    # f = norm_flow(f, 270, 480)

    # p1 = "/mnt/disks/dataset/Google_ois/test/s0_outdoor_panning_quick_VID_20200304_143946/frames/frame_0220.png"
    # p2 = "/mnt/disks/dataset/Google_ois/test/s0_outdoor_panning_quick_VID_20200304_143946/frames/frame_0221.png"

    # mesh = get_mesh(1, 270, 480)[0].cpu().numpy()
    # visualize_flow("flow.png", p1, p2, mesh, mesh+f)

    
