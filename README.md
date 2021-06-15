

# Deep Online Fused Video Stabilization

[[Paper]](https://arxiv.org/abs/2102.01279) [[Project Page]](https://zhmeishi.github.io/dvs/) [[Dataset]](https://storage.googleapis.com/dataset_release/all.zip) [[More Results]](https://zhmeishi.github.io/dvs/supp/results.html) 

This repository contains the Pytorch implementation of our method in the paper "Deep Online Fused Video Stabilization".

## Environment Setting
Python version >= 3.6
Pytorch with CUDA >= 1.0.0 (guide is [here](https://pytorch.org/get-started/locally/))
Install other used packages:
```
cd dvs
pip install -r requirements.txt --ignore-installed
```

## Data Preparation
Download sample video [here](https://drive.google.com/file/d/1nju9H8ohYZh6dGsdrQjQXFgfgkrFtkRi/view?usp=sharing). 
Uncompress the *video* folder under the *dvs* folder.

```
python load_frame_sensor_data.py 
```
Demo of curve visualization:
The **gyro/OIS curve visualization** can be found at *dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_real.jpg*.


## FlowNet2 Preparation
Note, we provide optical flow result of one test video in our Data Preparation. If you would like to generate them for all test videos, please follow [FlowNet2 official website](https://github.com/NVIDIA/flownet2-pytorch) and guide below. Otherwise, you can skip this section. 

Note, FlowNet2 installation is tricky. Please use Python=3.6 and Pytorch=1.0.0. More details are [here](https://github.com/NVIDIA/flownet2-pytorch/issues/156) or contact us for any questions.

Download FlowNet2 model *FlowNet2_checkpoint.pth.tar* [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view).  Move it under folder *dvs/flownet2*.
```
python warp/read_write.py # video2frames
cd flownet2
bash install.sh # install package
bash run.sh # generate optical flow file for dataset
``` 

## Running Inference 
```
python inference.py
python metrics.py
``` 
The loss and metric information will be printed in the terminal. The metric numbers can be slightly different due to difference on opencv/pytorch versions.  

The result is under *dvs/test/stabilzation*.   
In *s_114_outdoor_running_trail_daytime.jpg*, the blue curve is the output of our models, and the green curve is the input.   
*s_114_outdoor_running_trail_daytime_stab.mp4* is uncropped stabilized video.  
*s_114_outdoor_running_trail_daytime_stab_crop.mp4* is cropped stabilized video. Note, the cropped video is generated after running the metrics code.   

## Training
TBA

## Citation 
If you use this code or dataset for your research, please cite our paper.
```
@article{shi2021deep,
  title={Deep Online Fused Video Stabilization},
  author={Shi, Zhenmei and Shi, Fuhao and Lai, Wei-Sheng and Liang, Chia-Kai and Liang, Yingyu},
  journal={arXiv preprint arXiv:2102.01279},
  year={2021}
}
``` 
