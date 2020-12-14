# Distributed Dynamic Map Fusion in CARLA Simulation 
> the full version will be available after paper acceptance...

<img src="./preview/test2.png" width = "400" alt="图片名称" align=center />
<img src="./preview/test1.png" width = "400" alt="图片名称" align=center />

## Feature
- LiDAR/camera raw data collection in multi-agent synchronously
- data transformation to KITTI format
- data visualization with Open3d library

## Requirements

- Python3.6+
- open3d >= 0.10.0
- CARLA >= 0.9.8
- `apt install libxerces-c-3.2 libjpeg8`

## Install
1.  clone the repo

     ```
     git clone https://github.com/zijianzhang/CARLA_INVS.git
     ```

     > > remember change `CARLA_PATH` in `params.py`
     >
     > 1. 
     >
     > (suggest: `~/CARLA`)

2. enter the directory `CARLA_INVS` & install packages
     ```
     pip3 install -r requirement.txt
     ```

## Start
1. start up `CarlaUE4.sh`, and run the following script in shell to look for vehicles spawn points as Fig.1.

   ```
   python3 Scenario.py spawn
   ```

   <img src="./preview/carla.png" width = "250" height = "250"  alt="图片名称" align=center /> <img src="./preview/fig2.png" width = "250" alt="图片名称" align=center />

2. run the following script in shell to generate mulit-agent raw data. 

   ```bash
   python3 Scenario.py record x1,x2 y1,y2
   # The x1,x2,y1,y2 are the spawn points ID.
   ```

   where $\mathbf{x}=[x_1,...,x_N]$ for *human-driven vehicles*, $\mathbf{y}=[y_1,...,y_M]$ for *autonomous vehicles*.

3. run the following script in shell to transform raw data to KITTI format.

   ```bash
   python3 Process.py tmp/record2020_xxxx_xxxx
   ```

4. (Optional) run the following script in shell to view kitti Format data with Open3D as follows.

   ```bash
   python3 Visualization.py dataset/record2020_xxxx_xxxx vehicle_id frame_id
   # The vehicle_id is the intelligent vehicle ID. The frame_ID is the index of dataset.
   ```

<img src="./preview/fig3.png" width = "250" alt="图片名称" align=center />

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

label is the directory to save the ground truth labels.

calib is the calibration matrix from point cloud to image.

## Training Process
- training for pre-trained model
- training for federated model
- online distill training

## Testing Process
- with fusion
- without fusion