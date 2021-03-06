#! /usr/bin/env python3
import os
import subprocess as sp
import torch
import pdb
import copy
import time
from sys import argv
from averaging import average_weights
import numpy as np

ITER = int(argv[1])

try:
    os.remove('model/model713.pth')
    os.remove('model/model714.pth')
except Exception as e: 
    pass


if ITER == 0: # no pretrain model
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/tesla713.yaml --batch_size 1 --workers 0 --epochs 1']) #return model0
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/tesla714.yaml --batch_size 1 --workers 0 --epochs 1']) #return model1

if ITER >= 1: # train from last global model
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/tesla713.yaml --batch_size 1 --workers 0 --epochs 1 --pretrained_model ../model/model713.pth']) #return model0
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/tesla714.yaml --batch_size 1 --workers 0 --epochs 1 --pretrained_model ../model/model714.pth']) #return model1
  
# save model to dictionary
model_dict0 = torch.load('./model/checkpoint_epoch_0.pth'); params0 = model_dict0['model_state']
model_dict1 = torch.load('./model/checkpoint_epoch_1.pth'); params1 = model_dict1['model_state']

# append parameters into a list
w_locals = []
w_locals.append(params0)
w_locals.append(params1)

# compute perfect global model
params_avg = average_weights(w_locals)

# remove the model files
del(params0)
del(params1)
os.remove('model/checkpoint_epoch_0.pth')
os.remove('model/checkpoint_epoch_1.pth')

# save global model at each vehicle
torch.save({'model_state':params_avg,
            'optimizer_state': model_dict0['optimizer_state'],
            'accumulated_iter': model_dict0['accumulated_iter']
            },'model/model713.pth')

torch.save({'model_state':params_avg,
            'optimizer_state': model_dict1['optimizer_state'],
            'accumulated_iter': model_dict1['accumulated_iter']
            },'model/model714.pth')

# finish one FL iteration
print('---------------------------------FL Iteration %r is completed.'%argv[1])

