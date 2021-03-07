#!/usr/bin/env python3

# ============================================================================================= #
# Distributed dynamic map fusion via federated learning for intelligent networked vehicles,     #
# 2021 International Conference on Robotics and Automation (ICRA)                               #
# ============================================================================================= #

# amend relative import
import sys
from pathlib import Path
sys.path.append( Path(__file__).resolve().parent.parent.as_posix() ) #repo path
sys.path.append( Path(__file__).resolve().parent.as_posix() ) #file path
from params import *
# original import
import subprocess as sp
from pathlib import Path

ITER_MAX = 10
counter = 0
print(ENTRY_FILE)

while counter < ITER_MAX:
    print('===================== %d ====================='%counter)
    sp.run([ENTRY_FILE, counter])
    counter += 1
    pass
