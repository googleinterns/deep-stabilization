import numpy as np
import cv2
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

def load_video(path): # N x H x W x C
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
        # cv2.imwrite("result/frame%d_r.jpg" % count, image) 
        frames.append(image)
        success,image = vidcap.read()
        count += 1
        # if count > 4:
        #     break
    print("Video length: ", len(frames))
    return frames, fps, size

def save_video(path,frame_array, fps, size, frame_number = True):
    if path[-3:] == "mp4":
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    else:
        out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        if frame_number:
            frame_array[i] = draw_number(np.asarray(frame_array[i]), i)
        out.write(frame_array[i])
    out.release()

def draw_number(frame, num):
    image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./data/arial.ttf", 45)
     
    (x, y) = (10, 10)
    message = "Frame: " + str(num)
    color = 'rgb(0, 0, 0)' # black color
    
    draw.text((x, y), message, fill=color, font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def compare_difference(path1, path2):
    frames1, fps, size = load_video(path1)
    frames2, _, _ = load_video(path2)
    length = min(len(frames1),len(frames2))
    frames1 = np.array(frames1[:length]).astype(int)
    frames2 = np.array(frames2[:length]).astype(int)

    diff = np.abs(frames1 - frames2)
    # print(np.sum(diff))
    diff = diff.astype("uint8")
    save_video("./result/diff.mp4",diff, fps, size)


if __name__ == "__main__":
    mesh_data = load_mesh("./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt")
    # frame_array, fps, size = load_video("./data/testdata/results/s2_outdoor_runing_forward_VID_20200304_144434_feature.mp4")
    # save_video("./out.mp4",frame_array, fps, size)
    # p1 = "./result/output3.mp4"
    # p2 = "./data/testdata//results_full_identity/s2_outdoor_runing_forward_VID_20200304_144434_stab.mp4"
    # p3 = "./data/testdata/inputs/s2_outdoor_runing_forward_VID_20200304_144434/s2_outdoor_runing_forward_VID_20200304_144434.mp4"
    # compare_difference(p1,p2)
    # p4 = "/home/zhmeishi_google_com/dataset/Google/test/s1_outdoor_walking_forward_slerp_VID_20200304_143652/s1_outdoor_walking_forward_slerp_VID_20200304_143652.mp4"
    # frames, fps, size = load_video(p4)
    # print(np.array(frames).shape)
    # save_video("s2s.avi",frames,fps,size)
    result_poses_name = './data/testdata/results_updated/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt'
    result_poses = LoadStabResult(result_poses_name)
    grid1 = mesh_data["warping grid"]
    grid2 = np.reshape(result_poses['warping grid'],(-1,12,12,4))
    print(np.sum(np.abs(grid1 - grid2)))
    print(np.sum(np.abs(grid1) + np.abs(grid2)))

    
