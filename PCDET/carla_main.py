#!/usr/bin/env python3
import subprocess as sp

counter = 0
while counter <= 1:
    print('===================== %d ====================='%counter)
    sp.run(['./carla_fed.py', '%d'%counter])
    counter += 1