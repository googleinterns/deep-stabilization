# Deep Stabilization

This repository contains explorations on CNN-based video stabilization.
## Environment setting
Need GPU. 
Need to run flownet2-pytorch. See detail here:
[https://github.com/NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)

## Prepare Data
```
bash prepare_data.sh
``` 
## Prepare FlowNet
Download FlowNet2 here:
[https://github.com/NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)
```
# Go to flownet2 folder
bash install.sh # install the model
bash run.sh # generate optical flow file for dataset
``` 
## Run inference 
```
python inference.py # result in folder named test
``` 
## Run training 
```
python train.py # trained model in folder named checkpoint
``` 