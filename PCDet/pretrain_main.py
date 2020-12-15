#!/usr/bin/env python3
import os
import subprocess as sp
import torch
from averaging import average_weights
import pdb
import copy
import time
from sys import argv


sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain21.yaml --batch_size 1 --workers 0 --epochs 5"])
sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain31.yaml --batch_size 1 --workers 0 --epochs 5 --pretrained_model ../model/checkpoint_epoch_0.pth"])
sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain11.yaml --batch_size 1 --workers 0 --epochs 5 --pretrained_model ../model/checkpoint_epoch_1.pth"])

sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain22.yaml --batch_size 1 --workers 0 --epochs 15 --pretrained_model ../model/checkpoint_epoch_2.pth"])
sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain32.yaml --batch_size 1 --workers 0 --epochs 15 --pretrained_model ../model/checkpoint_epoch_3.pth"])
sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain12.yaml --batch_size 1 --workers 0 --epochs 15 --pretrained_model ../model/checkpoint_epoch_4.pth"])

sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain23.yaml --batch_size 1 --workers 0 --epochs 10 --pretrained_model ../model/checkpoint_epoch_5.pth"])
sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain33.yaml --batch_size 1 --workers 0 --epochs 10 --pretrained_model ../model/checkpoint_epoch_6.pth"])
sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/pretrain13.yaml --batch_size 1 --workers 0 --epochs 5 --pretrained_model ../model/checkpoint_epoch_7.pth"])






