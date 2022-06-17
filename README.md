# Distributed Dynamic Map Fusion and Federated Learning in CARLA Simulation
<img src="./preview/test2.png" width = "400" alt="图片名称" align=center /><img src="./preview/test1.png" width = "400" alt="图片名称" align=center />



[TOC]

# News

**INVS 2.0 (i.e., CarlaFLCAV) is now Available !!**

For further information, please follow this repo: https://github.com/SIAT-INVS/CarlaFLCAV

## Citation

```latex

@inproceedings{INVS,
  title={Distributed dynamic map fusion via federated learning for intelligent networked vehicles},
  author={Zijian Zhang and Shuai Wang and Yuncong Hong and Liangkai Zhou and Qi Hao},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}


@article{EFL,
  title={Edge federated learning via unit-modulus over-the-air computation},
  author={Shuai Wang and Yuncong Hong and Rui Wang and Qi Hao and Yik-Chung Wu and Derrick Wing Kwan Ng},
  journal={IEEE Transactions on Communications},
  year={2022},
  publisher={IEEE}
}

@article{FLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Qi Hao and Chengzhong Xu and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor},
  journal={arXiv preprint arXiv:2206.01748},
  year={2022}
}

```

## Dependency

- Ubuntu 18.04
- Python 3.7+
- CARLA >= 0.9.8, <=0.9.10
- CUDA>=10.0
- pytorch<=1.4.0
- llvm>=10.0

## Installation

1. Clone this repository to your workspace

      ```
     git clone https://github.com/lasso-sustech/CARLA_INVS.git --branch=main --depth=1
     ```

2. Enter the directory "CARLA_INVS" and install dependencies with `make`
     ```bash
     make dependency
     ```
     
     > It uses `apt` and `pip3` with network access. You can try speed up downloading with fast mirror sites.
     
3.  Download and extract the CARLA simulator somewhere (e.g., `~/CARLA`), and update `CARLA_PATH` in `params.py` with **absolute path** to the CARLA folder location.

> This repository is composed of three components: `gen_data` for dataset generation and visualization, `PCDet` for training and testing, `fusion` for global map fusion and visualization.
>
> The three components share the same configuration file `params.py`.



## Custom Dataset Generation
> **Features**: 1) LiDAR/camera raw data collection in multi-agent synchronously; 2) data transformation to KITTI format; 3) data visualization with Open3d library.

> **Tuning**: Tune the configurations as you like in `params.py` file under the `gen_data` section.

1. start up `CarlaUE4.sh` in your `CARLA_PATH` firstly and run the following script in shell to look for vehicles spawn points with *point_id*.

   ```bash
   python3 gen_data/Scenario.py spawn
   ```

   <img src="./preview/carla.png" width = "250" height = "250"  alt="图片名称" align=center /> <img src="./preview/fig2.png" width = "250" alt="图片名称" align=center />

2. run the following script to generate multi-agent raw data

   ```bash
   python3 gen_data/Scenario.py record [x_1,...x_N] [y_1,...y_M]
   ```

   where `x_1,...,x_N` is list of *point_ids* (separated by comma) for *human-driven vehicles*, and `y_1,...,y_M` for *autonomous vehicles* with sensors installation.

   The recording process would stop when `Ctrl+C` triggered, and the generated *raw data* will be put at `$ROOT_PATH/raw_data`.

3. Run the following script to transform raw data to KITTI format

   ```bash
   python3 gen_data/Process.py raw_data/record2020_xxxx_xxxx
   ```

   and the cooked KITTI format data will be put at `$ROOT_PATH/dataset`

4. (Optional) run the following script to view *KITTI Format data sample* with Open3D

   ```bash
   # The vehicle_id is the intelligent vehicle ID, and the frame_ID is the index of dataset.
   python3 gen_data/Visualization.py dataset/record2020_xxxx_xxxx vehicle_id frame_id
   ```

<img src="./preview/fig3.png" width = "250" alt="图片名称" align=center />

## Training Procedures

### Training for federated model

1. prepare sample dataset in `$ROOT_PATH/data` ([link](https://cloud.189.cn/t/jQJvuimquaEj))
2. run `python3 PCDet/INVS_main.py`

### Training for federated distill

> To be updated.



## Evaluation Procedures

### View local map
```bash
cd fusion;
python3 visualization/Visualization_local.py ../data/record2020_1027_1957 713 39336
```

### View fusion map
```bash
cd fusion;
python3 visualization/Visualization_fusion_map.py ../data/record2020_1027_1957 38549
```

## Contact

Should you have any question, please create issues, or contact [Shuai Wang](mailto:wangs3__at__sustech.edu.cn).

## Appendix

### Raw Data Format

````
tmp
   +- record2020_xxxx_xxxx
      +- label                #tmp labels
      +- vhicle.xxx.xxx_xxx
          +- sensor.camera.rgb_xxx
              +- 0000.jpg
              +- 0001.jpg
          +- sensor.camera.rgb_xxx_label
              +- 0000.txt
              +- 0001.txt
          +- sensor.lidar.rgb_cast_xxx
              +- 0000.ply
              +- 0001.ply
      +- vhicle.xxx.xxx_xxx
````

label is the directory to save the tmp labels.

### KITTI Format

````
dataset
   +- record2020_xxxx_xxxx
      +- global_label          #global labels
      +- vhicle.xxx.xxx_xxx
          +- calib00
              +- 0000.txt
              +- 0001.txt
          +- image00
              +- 0000.jpg
              +- 0001.jpg
          +- label00
              +- 0000.txt
              +- 0001.txt
          +- velodyne
              +- 0000.bin
              +- 0001.bin
      +- vhicle.xxx.xxx_xxx
````

- label is the directory to save the ground truth labels.

- calib is the calibration matrix from point cloud to image.

### PCDet Format

````
data
   +- record2020_xxxx_xxxx
      +- global_label    # same as global labels in “dataset”
      +- vhicle.xxx.xxx_xxx
   +- Imagesets
       +- train.txt   # same format as img_list.txt in “dataset”
       +- test.txt
       +- val.txt
   +- training
      +- calib  # same as calib00 in “dataset”
          +- 0000.txt
          +- 0001.txt
      +- image_2   # same as image00 in “dataset”
          +- 0000.jpg
          +1 0001.jpg
      +- label_2   # same as label00 in “dataset”
          +- 0000.txt
          +- 0001.txt
      +- velodyne   # same as velodyne in “dataset”
          +- 0000.bin
          +- 0001.bin
````

With the above data structure, run the following command
```
cd ./PCDet/pcdet/datasets/kitti
python3 preprocess.py create_kitti_infos record2020_xxxx_xxxx vhicle_id
```



### Authors

Zijian Zhang

[Yuncong Hong](https://iamhyc.github.io)

[Shuai Wang](https://faculty.sustech.edu.cn/wangs3/en/)
