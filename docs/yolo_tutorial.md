## Install dependencies
```
pip3 install opencv-python
pip3 install -r extra_modules/yolov5/requirements.txt
```

## Collect data
```
# Check spawn points
python3 dataset_tools/scenario.py spawn

# Start record raw_data
python3 dataset_tools/scenario.py record 2 31,25
```

## Generate yolo label
e.g. record_id = record2022_0110_1429
```
python3 dataset_tools/yolo_label.py --record_id=record2022_0110_1429
```

## Train yolov5
```
python3 extra_modules/yolov5/train.py --img 640 --batch 8 --epochs 100 --data raw_data/record2022_0110_1429/vehicle.tesla.model3_357/yolo_coco_carla.yaml --cfg extra_modules/yolov5/models/yolov5s.yaml  --weights yolov5s.pt
```

## Test Result
```
python3 yolov5/detect.py --source 'raw_data/record2022_0110_1429/vehicle.tesla.model3_358/yolo_dataset/images/train/*.jpg' --weights extra_modules/yolov5/runs/train/exp2/weights/best.pt
```