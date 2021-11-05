#!/usr/bin/env python3
from pathlib import Path
from os import chdir
import os

# Get CARLA Root Path
CARLA_PATH = os.environ.get('CARLA_ROOT')
if len(CARLA_PATH) == 0:
    CARLA_PATH = CARLA_PATH = os.environ.get('CARLA_PATH')
    if len(CARLA_PATH) == 0:
        CARLA_PATH = Path('~/carla').expanduser()
print("CARLA_PATH: {}".format(CARLA_PATH))

# Project Root Path
ROOT_PATH  = Path(__file__).parent
print("Project Root PATH: {}".format(ROOT_PATH))
##for gen_data
LOG_PATH   = ROOT_PATH / 'log'
RAW_DATA_PATH  = ROOT_PATH / 'raw_data'
COOK_DATA_PATH = ROOT_PATH / 'dataset'
#https://carla.readthedocs.io/en/latest/core_map/#changing-the-map
TOWN_MAP = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05'][1]
RAW_DATA_START  = 60#frame
RAW_DATA_END    = 180#frame
RAW_DATA_FREQ   = 1#Hz
RAW_DATA_FREQ_ALT=1#Hz, for img_list generation

##for PCDet
ENTRY_FILE = str(ROOT_PATH / 'PCDet' / 'fed.py')
TRAIN_ROOT = ROOT_PATH / 'PCDet' / 'tools'
MODEL_FOLDER = ROOT_PATH / 'model'



##utility
class workSpace:
    def __init__(self, p, *p_l, **kargs):
        self.wrk = Path(p, *p_l).expanduser().resolve()
        self.pwd = Path.cwd()
        if 'forceUpdate' in kargs.keys():
            self.forceUpdate = True
        else:
            self.forceUpdate = False
        pass
    
    def __enter__(self):
        if not Path(self.wrk).is_dir():
            if self.forceUpdate:
                Path(self.wrk).mkdir(mode=0o755, parents=True, exist_ok=True)
            else:
                return self.__exit__(*sys.exc_info())
        else:
            pass
        chdir(self.wrk)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        chdir(self.pwd)
        if exc_tb: pass
        pass
    pass
