#!/usr/bin/env python3
import os
import subprocess as sp
import torch
from averaging import average_weights
import pdb
import copy
import time
from sys import argv

device = torch.device('cpu')
# sp.run(['bash', '-c', "cd tools; python3 train.py --cfg_file cfgs/bear1.yaml --batch_size 1 --workers 0 --epochs 2"]) #return model_0
params0 = torch.load('./model/PartA2_car.pth', map_location=device)['model_state']
params1 = torch.load('./model/l1.pth', map_location=device)['model_state']

w_locals = []
w_locals.append(params0)
w_locals.append(params1)
params_avg = average_weights(w_locals)
del(params0)
del(params1)

torch.save({'model_state':params_avg}, 'model/t.pth')
print('-------------------------------------------------------------------------------success!')
