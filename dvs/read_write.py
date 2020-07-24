import numpy as np
import cv2
import os
from gyro.gyro_io import LoadStabResult
from PIL import Image, ImageDraw, ImageFont

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

def load_video(path, save_dir = None): # N x H x W x C
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    print(image.shape)
    height, width, layers = image.shape
    size = (width,height)
    count = 0
    frames = []
    while success:  
        # if count> 140:
        #     break
        if save_dir != None:
            path = os.path.join(save_dir, "frame_" + str(count).zfill(4) + ".png")
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
        video_path = os.path.join(data_folder, data_name[i]+".mp4")
        frame_folder = os.path.join(data_folder, "frames")
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)
        load_video(video_path, save_dir = frame_folder)

def save_video(path,frame_array, fps, size, frame_number = True, losses = None):
    if path[-3:] == "mp4":
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    else:
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        if frame_number:
            frame_array[i] = draw_number(np.asarray(frame_array[i]), i)
        if losses is not None:
            frame_array[i] = draw_number(np.asarray(frame_array[i]), losses[i], x = 900, message = "Loss: ")
        out.write(frame_array[i])
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
    length = min(len(frames1),len(frames2))

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
    # (x, y) = (10, 10)
    for i in range(len(grid1)):
        draw.line(xy = (grid1[i],grid2[i]), fill = (255,0,0), width = 5)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image) 



if __name__ == "__main__":
    # mesh_data = load_mesh("./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt")
    # frame_array, fps, size = load_video("./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_feature.mp4")
    # save_video("./out.mp4",frame_array, fps, size)
    # p1 = "./result/output3.mp4"
    # p2 = "./test/sample_test/s2_outdoor_runing_forward_VID_20200304_144227_stab.mp4"
    # p3 = "./data/testdata/videos_with_zero_virtual_motion/results_no_rotation/s2_outdoor_runing_forward_VID_20200304_144227_stab.mp4"
    # compare_difference(p3,p2)
    # p4 = "/home/zhmeishi_google_com/dataset/Google/test/s1_outdoor_walking_forward_slerp_VID_20200304_143652/s1_outdoor_walking_forward_slerp_VID_20200304_143652.mp4"
    # frames, fps, size = load_video(p4)
    # print(np.array(frames).shape)
    # save_video("s2s.avi",frames,fps,size)
    # result_poses_name = './data/testdata/results_updated/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt'
    # result_poses = LoadStabResult(result_poses_name)
    # grid1 = mesh_data["warping grid"]
    # grid2 = np.reshape(result_poses['warping grid'],(-1,12,12,4))
    # print(np.sum(np.abs(grid1 - grid2)))
    # print(np.sum(np.abs(grid1) + np.abs(grid2)))
    # video2frame("/home/zhmeishi_google_com/dataset/Google/train")
    p1 = "/mnt/disks/dataset/Google/train/s2_outdoor_runing_forward_VID_20200304_144434/frames/frame_0000.png"
    p2 = "/mnt/disks/dataset/Google/train/s2_outdoor_runing_forward_VID_20200304_144434/frames/frame_0001.png"
    visialize("test.png",p1,p2, None, None)

    
