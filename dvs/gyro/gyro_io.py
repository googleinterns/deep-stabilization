import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.io as sio
from .gyro_function import (
    ProcessGyroData, QuaternionProduct, QuaternionReciprocal, 
    ConvertQuaternionToAxisAngle, FindOISAtTimeStamp, GetMetadata,
    GetProjections, GetVirtualProjection, GetForwardGrid,
    CenterZoom, GetGyroAtTimeStamp, get_static, ConvertAxisAngleToQuaternion,
    ConvertAxisAngleToQuaternion_no_angle, ConvertQuaternionToAxisAngle_no_angle
    )

def load_gyro_mesh(input_name):
    data = LoadStabResult(input_name)
    w, h  = data["vertex_grid_size"][0]
    data["warping grid"] = np.reshape(data["warping grid"],(-1,int(w),int(h),4))
    return data

def get_grid(static_options, frame_data, quats_data, ois_data, virtual_data, no_shutter = False):
    grid = []
    result_poses = {}
    result_poses['virtual pose'] = virtual_data
    for i in range(len(virtual_data)):
        metadata = GetMetadata(frame_data, i)
        real_projections = GetProjections(static_options, metadata, quats_data, ois_data, no_shutter = no_shutter)
        # print(real_projections[0])
        virtual_projection = GetVirtualProjection(static_options, result_poses, metadata, i) 
        # print(virtual_projection)
        grid.append(GetForwardGrid(static_options, real_projections, virtual_projection))
    grid = np.array(grid)
    zoom_ratio = 1 / (1 - 2 * static_options["cropping_ratio"])
    curr_grid = CenterZoom(grid, zoom_ratio)
    curr_grid = np.transpose(curr_grid,(0,3,2,1))
    return curr_grid

def get_rotations(frame_data, quats_data, ois_data, num_frames):
    quats = np.zeros((num_frames, 4)) 
    for i in range(num_frames):
        quats[i,:] = GetGyroAtTimeStamp(quats_data, frame_data[i,0])

    rotations = np.zeros((num_frames,3))
    lens_offsets = np.zeros((num_frames, 2)) 
    for i in range(num_frames):
        if i != 0:
            quat_dif = QuaternionProduct(quats[i,:], QuaternionReciprocal(quats[i-1,:])) 
            axis_dif_cur = ConvertQuaternionToAxisAngle_no_angle(quat_dif)
            rotations[i,:] = axis_dif_cur
        lens_offsets[i,:] = FindOISAtTimeStamp(ois_data, frame_data[i, 4])     

    return rotations, lens_offsets

def visual_rotation(rotations_real, lens_offsets_real, rotations_virtual, lens_offsets_virtual, rotations_virtual2, lens_offsets_virtual2, path):
    # figure('units','normalized','outerposition',[0 0 1 1])
    plt.clf()
    plt.figure(figsize=(8,8))
    
    plt.subplot(5,1,1)
    plt.plot(rotations_real[:,0], "g")
    plt.plot(rotations_virtual[:,0], "b")
    if rotations_virtual2 is not None:
        plt.plot(rotations_virtual2[:,0], "r")
    plt.ylim(-0.02, 0.02)
    plt.xlabel('gyro x')

    plt.subplot(5,1,2)
    plt.plot(rotations_real[:,1], "g")
    plt.plot(rotations_virtual[:,1], "b")
    if rotations_virtual2 is not None:
        plt.plot(rotations_virtual2[:,1], "r")
    plt.ylim(-0.02, 0.02)
    plt.xlabel('gyro y')

    plt.subplot(5,1,3)
    plt.plot(rotations_real[:,2], "g")
    plt.plot(rotations_virtual[:,2], "b")
    if rotations_virtual2 is not None:
        plt.plot(rotations_virtual2[:,2], "r")
    plt.ylim(-0.02, 0.02)
    plt.xlabel('gyro z')
    
    plt.subplot(5,1,4)
    plt.plot(lens_offsets_real[:,0], "g")
    plt.plot(lens_offsets_virtual[:,0], "b")
    if rotations_virtual2 is not None:
        plt.plot(lens_offsets_virtual2[:,0], "r")
    plt.xlabel('ois x')

    plt.subplot(5,1,5)
    plt.plot(lens_offsets_real[:,1], "g")
    plt.plot(lens_offsets_virtual[:,1], "b")
    if rotations_virtual2 is not None:
        plt.plot(lens_offsets_virtual2[:,1], "r")
    plt.xlabel('ois y')

    plt.savefig(path)
    return

def LoadOISData(ois_name):
    ois_log = np.loadtxt(ois_name)
    ois_log = ois_log[:, -3:]
    return ois_log

def LoadFrameData(frame_log_name):
    frame_data = np.loadtxt(frame_log_name)
    frame_data[:, [0,4]] = frame_data[:, [0,4]] - np.expand_dims(frame_data[:,1]/2, axis = 1)
    return frame_data


def LoadGyroData(gyro_log_name):
    raw_gyro_data = np.loadtxt(gyro_log_name) 
    raw_gyro_data[:,0] = raw_gyro_data[:,0] * 1000 
    raw_gyro_data = raw_gyro_data[:,[0, 2, 1, 3]]

    [_, quats_data]  = ProcessGyroData(raw_gyro_data) 
    quats_data = np.concatenate((raw_gyro_data[:, 0, None], quats_data), axis = 1)
    return quats_data

def LoadStabResult(input_name):
    fid = open(input_name)
    data = {}
    while True:
        name, val = ReadLine(fid)
        if name == None:
            break
        if name in data:
            data[name] = np.concatenate((data[name], val), axis=0)
        else:
            data[name] = val
    fid.close()
    print("Mesh length: ", len(list(data.values())[0]))
    return data


def ReadLine(fid):
    name = ''
    val = 0
    tline = fid.readline()
    if len(tline) == 0:
        return None, None
    if tline[-1] == "\n":
        tline = tline[:-1]
    ind = tline.find(':')
    name = tline[:ind]
    tmp_val= str2num(tline[ind+1:])
    if len(tmp_val) > 0:
        val = tmp_val
    else:
        tline = fid.readline()
        if tline[-1] == "\n":
            tline = tline[:-1]
        val = str2num(tline)
    return name, np.expand_dims(np.array(val), axis=0)

def str2num(string):
    nums = string.split(" ")
    nums = [float(_) for _ in nums if _ != ""]
    return nums
    

if __name__ == "__main__":
    result = '/home/zhmeishi_google_com/dvs/data/testdata/results_updated/s2_outdoor_runing_forward_VID_20200304_144434_stab_mesh.txt'
    LoadStabResult(result)
    