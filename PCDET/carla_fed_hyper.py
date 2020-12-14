#!/usr/bin/env python3
import os
import subprocess as sp
import torch
from averaging import average_weights
import pdb
import copy
import time
from sys import argv

if int(argv[1]) == 0:
    sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/train1_new.yaml --batch_size 1 --workers 0 --epochs 2"])
    sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/train2_new.yaml --batch_size 1 --workers 0 --epochs 2"])
    sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/train4_new.yaml --batch_size 1 --workers 0 --epochs 2"])
    sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/train5_new.yaml --batch_size 1 --workers 0 --epochs 2"])
    # sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/train6_new.yaml --batch_size 1 --workers 0 --epochs 2"])
    pass

if int(argv[1]) == 1:
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train1_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_0
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train2_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train4_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train5_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train6_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1

if int(argv[1]) == 2:
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train1_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_0
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train2_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train4_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train5_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train6_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    pass


if int(argv[1]) >= 3:
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train1_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_0
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train2_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train4_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train5_new.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    # sp.run(['bash', '-c', 'cd tools; python3 train.py --cfg_file cfgs/train6_new_slow.yaml --batch_size 1 --workers 0 --epochs 2 --pretrained_model ../model/model_avg.pth']) #return model_1
    # 0.00001 0.000005
    pass


model_dict0 = torch.load('./model/checkpoint_epoch_0.pth'); params0 = model_dict0['model_state']
model_dict1 = torch.load('./model/checkpoint_epoch_1.pth'); params1 = model_dict1['model_state']
model_dict2 = torch.load('./model/checkpoint_epoch_2.pth'); params2 = model_dict2['model_state']
model_dict3 = torch.load('./model/checkpoint_epoch_3.pth'); params3 = model_dict3['model_state']
if int(argv[1]) <= 2: model_dict4 = torch.load('./model/checkpoint_epoch_4.pth'); params4 = model_dict4['model_state']
# pdb.set_trace()

w_locals = []
w_locals.append(params0)
w_locals.append(params1)
w_locals.append(params2)
w_locals.append(params3)
if int(argv[1]) <= 2: w_locals.append(params4)
params_avg = average_weights(w_locals)

del(params0)
del(params1)
del(params2)
del(params3)
if int(argv[1]) <= 2: del(params4)
os.remove('model/checkpoint_epoch_0.pth')
os.remove('model/checkpoint_epoch_1.pth')
os.remove('model/checkpoint_epoch_2.pth')
os.remove('model/checkpoint_epoch_3.pth')
if int(argv[1]) <= 2: os.remove('model/checkpoint_epoch_4.pth')

torch.save({'model_state':params_avg,
            'optimizer_state': model_dict0['optimizer_state'],
            # 'epoch': model_dict0['epoch'],
            'accumulated_iter': model_dict0['accumulated_iter']
            },'model/model_avg.pth')

del(params_avg)


print('---------------------------------Average---------------------------------------------- %r.'%argv[1])

