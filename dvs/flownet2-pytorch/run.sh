#!/bin/bash
python main.py --inference --model FlowNet2 --save_flow --inference_dataset Google \
	--inference_dataset_root /mnt/disks/dataset/Google/test \
	--resume /home/zhmeishi_google_com/optical_flow/flownet2-pytorch/FlowNet2_checkpoint.pth.tar \
	--inference_visualize

python main.py --inference --model FlowNet2 --save_flow --inference_dataset Google \
	--inference_dataset_root /mnt/disks/dataset/Google/train \
	--resume /home/zhmeishi_google_com/optical_flow/flownet2-pytorch/FlowNet2_checkpoint.pth.tar \
	--inference_visualize
