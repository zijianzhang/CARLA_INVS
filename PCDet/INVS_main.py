# Distributed dynamic map fusion via federated learning for intelligent networked vehicles, 
# 2021 International Conference on Robotics and Automation (ICRA)

import subprocess as sp

ITER_MAX = 10
counter = 0
while counter < ITER_MAX:
    print('===================== %d ====================='%counter)
    sp.run(['./fed.py', '%d'%counter])
    counter += 1