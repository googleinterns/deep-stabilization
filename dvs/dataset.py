from torch.utils.data import Dataset
import os
import collections
from gyro import (
    LoadGyroData, 
    LoadOISData, 
    LoadFrameData, 
    GetGyroAtTimeStamp, 
    get_static, 
    GetMetadata, 
    GetProjections, 
    train_GetGyroAtTimeStamp,
    QuaternionProduct,
    QuaternionReciprocal,
    train_ConvertQuaternionToAxisAngle, 
    ConvertAxisAngleToQuaternion_no_angle
    )
import random
import numpy as np
import torchvision.transforms as transforms
import torch
from flownet2 import flow_utils
from scipy import ndimage, misc
from numpy import linalg as LA

def get_data_loader(cf):
    size = cf["data"]["batch_size"]
    num_workers = cf["data"]["num_workers"]
    train_data, test_data = get_dataset(cf)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=size,shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=size,shuffle=False, pin_memory=True, num_workers=num_workers)
    return trainloader,testloader

def get_dataset(cf):
    resize_ratio = cf["data"]["resize_ratio"]
    train_transform, test_transform = _data_transforms()
    train_path = os.path.join(cf["data"]["data_dir"], "train")
    test_path = os.path.join(cf["data"]["data_dir"], "test")

    train_data = Dataset_Gyro(
        train_path, sample_freq = cf["data"]["sample_freq"]*1000000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000000, transform = train_transform, resize_ratio = resize_ratio)
    test_data = Dataset_Gyro(
        test_path, sample_freq = cf["data"]["sample_freq"]*1000000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000000, transform = test_transform, resize_ratio = resize_ratio)
    return train_data, test_data

def get_inference_data_loader(cf, data_path, no_flo = False):
    test_data = get_inference_dataset(cf, data_path, no_flo)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False, pin_memory=True, num_workers=1)
    return testloader

def get_inference_dataset(cf, data_path, no_flo = False):
    resize_ratio = cf["data"]["resize_ratio"]
    _, test_transform = _data_transforms()
    test_data = Dataset_Gyro(
        data_path, sample_freq = cf["data"]["sample_freq"]*1000000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000000, transform = test_transform, resize_ratio = resize_ratio,
        inference_only = True, no_flo = no_flo)
    return test_data

def _data_transforms():

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    return train_transform, test_transform

class DVS_data():
    def __init__(self):
        self.gyro = None
        self.ois = None
        self.frame = None
        self.length = 0
        self.flo_path = None
        self.flo_shape = None
        self.flo_back_path = None
        self.static_options = get_static()

class Dataset_Gyro(Dataset):
    def __init__(self, path, sample_freq = 33*1000000, number_real = 10, time_train = 2000*1000000, \
        transform = None, inference_only = False, no_flo = False, resize_ratio = 1): 
        r"""
        Arguments:
            sample_freq: real quaternions [t-sample_freq*number_real, t+sample_freq*number_real] ns
            number_real: real gyro num in half time_interval
            time_train: time for a batch ns
        """
        self.sample_freq = sample_freq
        self.number_real = number_real
        self.no_flo = no_flo
        self.resize_ratio = resize_ratio
        self.inference_only = inference_only

        if inference_only:
            self.length = 1
            self.data = [self.process_one_video(path)]
            self.number_train = self.data[0].length
            return

        self.time_train = time_train
        self.number_train = time_train//self.sample_freq

        # self.number_train = 100
        
        self.data_name = sorted(os.listdir(path))
        self.length = len(self.data_name)
        self.data = []
        for i in range(self.length):
            self.data.append(self.process_one_video(os.path.join(path,self.data_name[i])))
    
    def process_one_video(self, path):
        dvs_data = DVS_data()
        files = sorted(os.listdir(path))
        print(path)
        for f in files:
            file_path = os.path.join(path,f)
            if "frame" in f and "txt" in f:
                dvs_data.frame = LoadFrameData(file_path)
                print("frame:", dvs_data.frame.shape, end="    ")
            elif "gyro" in f:
                dvs_data.gyro = LoadGyroData(file_path)
                # print("gyro:", dvs_data.gyro.shape, end="    ")
                dvs_data.gyro = preprocess_gyro(dvs_data.gyro)
                # gyro_time = dvs_data.gyro[:,0] 
                print("gyro:", dvs_data.gyro.shape, end="    ")
            elif "ois" in f:
                dvs_data.ois = LoadOISData(file_path)
                print("ois:", dvs_data.ois.shape, end="    ")
                # ois_time = dvs_data.ois[:,2] 
            elif f == "flo":
                dvs_data.flo_path, dvs_data.flo_shape = LoadFlow(file_path)
                print("flo_path:", len(dvs_data.flo_path), end="    ")
                print("flo_shape:", dvs_data.flo_shape, end="    ")
            elif f == "flo_back":
                dvs_data.flo_back_path, _ = LoadFlow(file_path)
            
        print()
        dvs_data.length = min(dvs_data.frame.shape[0] - 1, len(dvs_data.flo_path))
        dvs_data.static_options = get_static(height = dvs_data.flo_shape[0], width = dvs_data.flo_shape[1])
        return dvs_data

    def generate_quaternions(self, dvs_data):
        first_id = random.randint(0, dvs_data.length - self.number_train) + 1 # skip the first frame
        # ratio = random.randint(0, 100)
        # if ratio < 30:
        # first_id = 1

        sample_data = np.zeros((self.number_train, 2 * self.number_real + 1, 4))
        sample_time = np.zeros((self.number_train+1))
        sample_time[0] = dvs_data.frame[first_id - 1, 0]

        real_postion = np.zeros((self.number_train, 4))

        time_start = sample_time[0]

        for i in range(self.number_train):
            sample_time[i+1] = dvs_data.frame[first_id + i, 0]
            real_postion[i] = GetGyroAtTimeStamp(dvs_data.gyro, sample_time[i+1])
            for j in range(-self.number_real, self.number_real+1):
                index = j + self.number_real
                time_stamp = sample_time[i+1] + self.sample_freq * j 
                # if time_start >= time_stamp:
                #     sample_data[i, index] = get_data_at_timestamp(dvs_data.gyro, time_start, self.sample_freq)
                # else:
                sample_data[i, index] = get_data_at_timestamp(dvs_data.gyro, time_stamp, self.sample_freq)
                
        sample_data = np.reshape(sample_data, (self.number_train, (2*self.number_real+1) * 4))
        return sample_data, sample_time, first_id, real_postion

    def load_flo(self, idx, first_id):
        shape = self.data[idx].flo_shape
        # h, w = int(self.resize_ratio * shape[0]), int(self.resize_ratio * shape[1])
        h, w = shape[0], shape[1]
        flo = np.zeros((self.number_train, h, w, 2))
        flo_back = np.zeros((self.number_train, h, w, 2))

        for i in range(self.number_train):
            frame_id = i + first_id
            f = flow_utils.readFlow(self.data[idx].flo_path[frame_id-1]).astype(np.float32) 
            # f = resize_flow(f, self.resize_ratio) # Have done on the dataset
            # f = norm_flow(f, shape)
            flo[i] = f

            f_b = flow_utils.readFlow(self.data[idx].flo_back_path[frame_id-1]).astype(np.float32) 
            # f_b = resize_flow(f_b, self.resize_ratio)
            # f_b = norm_flow(f_b, shape)
            flo_back[i] = f_b

        return flo, flo_back

    def load_real_projections(self, idx, first_id):
        static_options = get_static() # TODO: May need to change
        real_projections = np.zeros((self.number_train + 1, static_options["num_grid_rows"], 3, 3))
        for i in range(self.number_train + 1):
            frame_id = i + first_id
            metadata = GetMetadata(self.data[idx].frame, frame_id - 1)
            # real_projections[i] = np.array(GetProjections(static_options, metadata, self.data[idx].gyro, self.data[idx].ois))
            real_projections[i] = np.array(GetProjections(static_options, metadata, self.data[idx].gyro, np.zeros(self.data[idx].ois.shape)))
        return real_projections

    def __getitem__(self, idx):
        inputs, times, first_id, real_postion = self.generate_quaternions(self.data[idx])
        if self.no_flo:
            return inputs, times 
        real_projections = self.load_real_projections(idx, first_id)
        flo, flo_back = self.load_flo(idx, first_id)
        return inputs, times, flo, flo_back, real_projections, real_postion, idx

    def __len__(self):
        return self.length

    def get_virtual_data(self, virtual_queue, real_queue_idx, pre_times, cur_times, time_start, batch_size, number_virtual, sample_freq):
        # virtual_queue: [batch_size, num, 5 (timestamp, quats)]
        # eular angle, 
        # deta R angular velocity [Q't-1, Q't-2] 
        # output virtual angular velocity, x, x*dtime => detaQt
        sample_freq *= 1000000
        virtual_data = np.zeros((batch_size, number_virtual, 4))
        vt_1 = np.zeros((batch_size, 4))
        for i in range(batch_size):
            sample_time = cur_times[i]
            for j in range(number_virtual):
                time_stamp = sample_time - sample_freq * (j+1) 
                virtual_data[i, -j-1] = get_virtual_at_timestamp(virtual_queue[i], self.data[real_queue_idx[i]].gyro, time_stamp, time_start[i])
            vt_1[i] = get_virtual_at_timestamp(virtual_queue[i], self.data[real_queue_idx[i]].gyro, pre_times[i], time_start[i])
        virtual_data = np.reshape(virtual_data, (batch_size, number_virtual * 4))
        return torch.tensor(virtual_data, dtype=torch.float), torch.tensor(vt_1, dtype=torch.float)

    def update_virtual_queue(self, batch_size, virtual_queue, out, times, real_position):
        virtual_data = np.zeros((batch_size, 5))
        virtual_data[:,0] = times
        virtual_data[:, 1:] = out
        for i in range(batch_size):
            # virtual_data[i,1:] = QuaternionProduct(real_position[i], ConvertAxisAngleToQuaternion(out[i,:3], out[i,3]))  
            # virtual_data[i,1:] = ConvertAxisAngleToQuaternion_no_angle(out[i,:3])
            virtual_data[i,1:] = out[i] 
        virtual_data = np.expand_dims(virtual_data, axis = 1)

        if None in virtual_queue:
            virtual_queue = virtual_data
        else:
            virtual_queue = np.concatenate((virtual_queue, virtual_data), axis = 1)
        return virtual_queue
    

    

def preprocess_gyro(gyro, extend = 200):
    fake_gyro = np.zeros((extend, 5))
    time_start = gyro[0,0]
    for i in range(extend):
        fake_gyro[-i-1, 0] = time_start - (gyro[i+1, 0] - time_start)
        fake_gyro[-i-1, 1:] = gyro[i+1, 1:]

    new_gyro = np.concatenate((fake_gyro, gyro), axis = 0)
    return new_gyro

def LoadFlow(path):
    file_names = sorted(os.listdir(path))
    file_path =[]
    for n in file_names:
        file_path.append(os.path.join(path, n))
    return file_path, flow_utils.readFlow(file_path[0]).shape

def resize_flow(flow, ratio):
    f0 = np.expand_dims(ndimage.zoom(flow[:,:,0], ratio), axis = 2)
    f1 = np.expand_dims(ndimage.zoom(flow[:,:,1], ratio), axis = 2)
    return np.concatenate((f0,f1),axis=2)

def get_data_at_timestamp(gyro_data, time_stamp, sample_freq = None):
    quat_t = GetGyroAtTimeStamp(gyro_data, time_stamp)
    # quat_t_1 = GetGyroAtTimeStamp(gyro_data, time_stamp - sample_freq)
    # quat_dif = QuaternionProduct(quat_t, QuaternionReciprocal(quat_t_1))  
    # return train_ConvertQuaternionToAxisAngle(quat_t) 
    return quat_t

def get_virtual_at_timestamp(virtual_queue, real_queue, time_stamp, time_start, sample_freq = None):
    # if time_stamp < time_start:
    #     return GetGyroAtTimeStamp(real_queue, time_start)
    if virtual_queue is None:
        quat_t = GetGyroAtTimeStamp(real_queue, time_stamp)
        # quat_t= np.array([0,0,0,1])
    else:
        quat_t = train_GetGyroAtTimeStamp(virtual_queue, time_stamp)
        if quat_t is None:
            quat_t = GetGyroAtTimeStamp(real_queue, time_stamp)
            # quat_t= np.array([0,0,0,1])
    return quat_t
    # return train_ConvertQuaternionToAxisAngle(quat_t) 

if __name__ == "__main__":
    path = "/mnt/disks/dataset/Google/test/s2_outdoor_runing_forward_VID_20200304_144434/flo"
    # Dataset_Gyro(path)
    LoadFlow(path)