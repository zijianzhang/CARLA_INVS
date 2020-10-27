# Multi-agent Data Collection Scripts in CARLA Simulation 
![screenshot](./doc/pcd_label.png)

## Feature
- LiDAR/camera raw data collection in multi-agent synchronously
- data format transformation to KITTI format
- data visualization with Open3d library


## Requirements

python>=3.6, open3d>=0.10.0, CARLA >=0.9.8 

## Install
1. install packages
     ```
     pip install -r requirement.txt
     ```
## Start
run the following script in shell to look for vehicles spawn points as Fig.1.
```
python Scenario.py spawn
```
run the following script in shell to generate mulit-agent raw data. 
```
python Scenario.py record x1,x2 y1,y2
# The x1,x2,y1,y2 are the spawn points ID.
```
run the following script in shell to transform raw data to KITTI format.
```
python Process.py tmp/record2020_xxxx_xxxx
```
run the following script in shell to view kitti Format data with open3D as Fig.2.

```
python Visualization.py tmp/record2020_xxxx_xxxx vehicle_id frame_id
# The vehicle_id is the intelligent vehicle ID. The frame_ID is the index of dataset.
```

## Raw Data Format

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

## KITTI Format

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
