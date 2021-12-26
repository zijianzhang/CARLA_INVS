## Install dependencies
```
pip3 install opencv-python
pip3 install -r yolov5/requirements.txt
```

## Collect data
```
python3 gen_data/Scenario.py spawn
python3 gen_data/Scenario.py record 36 11,57
```
## Generate yolo label
e.g. record_id = record2021_1106_0049
```
python3 yolo_label.py --record_id=record2021_1105_2340
```

## Train yolov5
```
 python3 yolov5/train.py --img 640 --batch 8 --epochs 100 --data raw_data/record2021_1106_0049/vehicle.tesla.model3_248/yolo_coco_carla.yaml --cfg yolov5/models/yolov5s.yaml  --weights yolov5s.pt
```

## Test Result
Test dataset = record2021_1106_0049 / vehicle.tesla.model3_248
```
python3 yolov5/detect.py --source 'raw_data/record2021_1106_0049/vehicle.tesla.model3_248/yolo_dataset/images/train/*.jpg' --weights yolov5/runs/train/exp6/weights/best.pt 
```