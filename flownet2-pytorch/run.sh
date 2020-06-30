#!/bin/bash
python main.py --inference --model FlowNet2 --inference_dataset Google \
	--inference_dataset_root /path/to/dataset/Google/train \
	--resume /path/to/flownet2-pytorch/FlowNet2_checkpoint.pth.tar \
	--inference_visualize --save_flow

python main.py --inference --model FlowNet2 --inference_dataset Google \
	--inference_dataset_root /path/to/dataset/Google/test \
	--resume /path/to/flownet2-pytorch/FlowNet2_checkpoint.pth.tar \
	--inference_visualize --save_flow
