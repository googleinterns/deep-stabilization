
# Deep Online Fused Video Stabilization

This repository contains the Pytorch implementation of [Deep Online Fused Video Stabilization](https://arxiv.org/abs/2102.01279). See more video results [here](https://zhmeishi.github.io/dvs/).
## Environment setting
Need GPU and Pytorch.

## Prepare Data
Download sample video [here](https://drive.google.com/file/d/1nju9H8ohYZh6dGsdrQjQXFgfgkrFtkRi/view?usp=sharing) (need to requset). 

After uncompress, move the "video" folder under the "dvs" folder. 

```
cd dvs
python rm_ois.py # generete video used in metrics.py 
```
Demo of curve visualization and mesh-based-warping:

The **rolling shutter correction without stabilization result** can be found at "dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_no_shutter.mp4".

The **gyro/OIS curve visualization** can be found at "dvs/video/s_114_outdoor_running_trail_daytime/ControlCam_20200930_104820_real.jpg".


## Prepare FlowNet2
Note, we provide optical flow result in our data. If you would like to generate them by yourself, please follow [FlowNet2 official website](https://github.com/NVIDIA/flownet2-pytorch) and guide below. Otherwise, you can skip this section. 

Download FlowNet2 model "FlowNet2_checkpoint.pth.tar" [here](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view).

Move "FlowNet2_checkpoint.pth.tar" under folder "dvs/flownet2".
```
python warp/read_write.py # video2frames
cd flownet2
bash install.sh # install package
bash run.sh # generate optical flow file for dataset
``` 
## Run inference 
```
python inference.py
python metrics.py
``` 
The result is under "dvs/test/iccv_6". 

In **s_114_outdoor_running_trail_daytime.jpg**, blue curve is the output of our models and green curve is input. 

**s_114_outdoor_running_trail_daytime_stab.mp4** is uncropped stabilized video and **s_114_outdoor_running_trail_daytime_stab_crop.mp4** is cropped stabilized video. Note the cropped video is generated after running the metrics code. 
