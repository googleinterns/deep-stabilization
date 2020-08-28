#!/bin/bash
echo "generate video without ois"
python rm_ois.py
python warp/read_write.py