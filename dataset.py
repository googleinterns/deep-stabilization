from torch.utils.data import Dataset
import os
import collections
from gyro import LoadGyroData, LoadOISData, LoadFrameData
from gyro import GetGyroAtTimeStamp
import random
import numpy as np
import torchvision.transforms as transforms
import torch

def get_data_loader(cf):
    size = cf["data"]["batch_size"]
    num_workers = cf["data"]["num_workers"]
    train_data, test_data = get_dataset(cf)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=size,shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=size,shuffle=False, pin_memory=True, num_workers=num_workers)
    return trainloader,testloader

def get_dataset(cf):
    train_transform, test_transform = _data_transforms()
    train_path = os.path.join(cf["data"]["data_dir"], "train")
    test_path = os.path.join(cf["data"]["data_dir"], "test")

    train_data = Dataset_Gyro(
        train_path, sample_freq = cf["data"]["sample_freq"]*1000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000, transform = train_transform)
    test_data = Dataset_Gyro(
        test_path, sample_freq = cf["data"]["sample_freq"]*1000, number_real = cf["data"]["number_real"], 
        time_train = cf["data"]["time_train"]*1000, transform = test_transform)
    return train_data, test_data

def get_inference_data_loader(cf, data_path):
    test_data = get_inference_dataset(cf, data_path)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False, pin_memory=True, num_workers=1)
    return testloader

def get_inference_dataset(cf, data_path):
    _, test_transform = _data_transforms()
    test_data = Dataset_Gyro(data_path, transform = test_transform, inference_only = True)
    return test_data

def _data_transforms():

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    return train_transform, test_transform

def get_virtual_data(virtual_queue, times, batch_size, number_virtual, sample_freq):
    # eular angle, 
    # deta R angular velocity [Q't-1, Q't-2] 
    # output virtual angular velocity, x, x*dtime => detaQt
    virtual_data = np.zeros((batch_size, number_virtual, 4))
    if virtual_queue is None:
        virtual_data[:,:,3] = 1
    else:
        for i in range(batch_size):
            sample_time = times[i]
            for j in range(number_virtual):
                time_stamp = sample_time - sample_freq * j # decreasing order
                virtual_data[i, j] = GetGyroAtTimeStamp(virtual_queue[i], time_stamp)
    virtual_data = np.reshape(virtual_data, (batch_size, number_virtual * 4))
    return torch.tensor(virtual_data, dtype=torch.float)

    
class Gyro_data():
    def __init__(self):
        self.gyro = None
        self.ois = None
        self.frame = None
        self.length = 0

class Dataset_Gyro(Dataset):
    def __init__(self, path, sample_freq = 33*1000, number_real = 10, time_train = 2000*1000, \
        transform = None, inference_only = False): 
        r"""
        Arguments:
            sample_freq: real quaternions [t-sample_freq*number_real, t+sample_freq*number_real] ns
            number_real: real gyro num in half time_interval
            time_train: time for a batch ns
        """
        # if transform is None:
        #     self.transfrom = transforms.Compose([
        #         transforms.ToTensor(),
        #     ])
        # else:
        #     self.transfrom = transform

        self.sample_freq = sample_freq
        self.number_real = number_real

        if inference_only:
            self.length = 1
            self.data = [self.process_one_video(path)]
            self.number_train = self.data[0].length 
            return

        self.time_train = time_train
        self.number_train = time_train//self.sample_freq
        
        self.data_name = sorted(os.listdir(path))
        self.length = len(self.data_name)
        self.data = []
        for i in range(self.length):
            self.data.append(self.process_one_video(os.path.join(path,self.data_name[i])))
    
    def process_one_video(self, path):
        gyro_data = Gyro_data()
        files = sorted(os.listdir(path))
        print(path)
        for f in files:
            file_path = os.path.join(path,f)
            if "frame" in f:
                gyro_data.frame = LoadFrameData(file_path)
                print("frame:", gyro_data.frame.shape, end="    ")
            elif "gyro" in f:
                gyro_data.gyro = LoadGyroData(file_path)
                # gyro_time = gyro_data.gyro[:,0] 
                print("gyro:", gyro_data.gyro.shape, end="    ")
            elif "ois" in f:
                gyro_data.ois = LoadOISData(file_path)
                print("ois:", gyro_data.ois.shape, end="    ")
                # ois_time = gyro_data.ois[:,2] 
        print()
        gyro_data.length = min(gyro_data.frame.shape[0], gyro_data.gyro.shape[0], gyro_data.ois.shape[0])
        return gyro_data

    def generate_quaternions(self, gyro_data):
        first_id = random.randint(0, gyro_data.length - self.number_train)
        sample_data = np.zeros((self.number_train, 2 * self.number_real + 1, 4))
        sample_time = np.zeros((self.number_train))
        for i in range(self.number_train):
            sample_time[i] = gyro_data.gyro[first_id + i, 0]
            for j in range(-self.number_real, self.number_real+1):
                time_stamp = sample_time[i] - self.sample_freq * j # decreasing order
                sample_data[i, j] = GetGyroAtTimeStamp(gyro_data.gyro, time_stamp)
        sample_data = np.reshape(sample_data, (self.number_train, (2*self.number_real+1) * 4))
        return sample_data, sample_time

    def __getitem__(self, idx):
        inputs, times = self.generate_quaternions(self.data[idx])
        return inputs, times

    def __len__(self):
        return self.length

if __name__ == "__main__":
    path = "/home/zhmeishi_google_com/dataset/Google/train"
    Dataset_Gyro(path)