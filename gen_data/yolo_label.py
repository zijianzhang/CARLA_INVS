#!/usr/bin/python3
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # repo path
sys.path.append(Path(__file__).resolve().parent.as_posix())  # file path

from params import *
import os
import cv2
import numpy as np
import glob
import yaml
import argparse

TRAFFIC_LIGHT_SEG_COLOR = (250, 170, 30)


class LABEL_ID:
    TRAFFIC_LIGHT = 9
    TRAFFIC_LIGHT_RED = 80
    TRAFFIC_LIGHT_YELLOW = 81
    TRAFFIC_LIGHT_GREEN = 82


class YoloLabel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_output_path = Path(data_path) / "../yolo_dataset"
        self.data_output_path = Path(os.path.abspath(self.data_output_path.as_posix()))
        print(self.data_output_path)
        self.image_out_path = self.data_output_path / "images" / "train"
        self.label_out_path = self.data_output_path / "labels" / "train"
        self.image_rgb = None
        self.image_seg = None
        self.preview_img = None
        self.rec_pixels_min = 160
        self.color_pixels_min = 40

    def process(self):
        img_path_list = sorted(glob.glob(self.data_path + '/*.png'))
        img_seg_path_list = sorted(glob.glob(self.data_path + '/seg' + '/*.npz'))
        for rgb_img, seg_img in zip(img_path_list, img_seg_path_list):
            self.label_img(rgb_img, seg_img)

    def label_img(self, rgb_img_path, seg_img_path):
        success = self.check_id(rgb_img_path, seg_img_path)
        if not success:
            return
        self.image_rgb = None
        self.image_seg = None
        self.image_rgb = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        self.image_seg = (np.load(seg_img_path))['a']
        if self.image_rgb is None or self.image_seg is None:
            return
        self.image_rgb = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGBA2RGB)
        self.image_seg = cv2.cvtColor(self.image_seg, cv2.COLOR_BGRA2RGB)
        img_name = os.path.basename(rgb_img_path)
        height, width, _ = self.image_rgb.shape

        mask = (self.image_seg == (250, 170, 30))
        tmp_mask = (mask.sum(axis=2, dtype=np.uint8) == 3)
        mono_img = np.array(tmp_mask * 255, dtype=np.uint8)

        self.preview_img = self.image_rgb
        # self.preview_img = self.image_seg
        # cv2.imshow("seg", self.preview_img)
        # cv2.imshow("mono", mono_img)
        # cv2.waitKey()
        contours, _ = cv2.findContours(mono_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        labels = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self.rec_pixels_min:
                continue
            # cv2.rectangle(self.preview_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.imshow("rect", self.preview_img)
            # cv2.waitKey()
            max_y, max_x, _ = self.image_rgb.shape
            if y + h >= max_y or x + w >= max_x:
                continue

            color = self.check_color(self.image_rgb[y:y + h, x:x + w, :])

            if color is not None:
                # cv2.putText(self.preview_img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                # cv2.rectangle(self.image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
                label_info = "{} {} {} {} {}".format(color,
                                                     float(x + (w / 2.0)) / width,
                                                     float(y + (h / 2.0)) / height,
                                                     float(w) / width,
                                                     float(h) / height)
                labels.append(label_info)
                # cv2.imshow("result", self.preview_img)
                # cv2.imshow("test", self.preview_img[y:y+h, x:x+w, :])
                # cv2.waitKey()

        if len(labels) > 0:
            os.makedirs(self.label_out_path, exist_ok=True)
            os.makedirs(self.image_out_path, exist_ok=True)
            print("image\t\t\twidth\theight\n{}\t{}\t{}".format(img_name, width, height))
            print("Got {} labels".format(len(labels)))
            cv2.imwrite(self.image_out_path.as_posix() + '/' + os.path.splitext(img_name)[0] + '.jpg', self.image_rgb)
            print(self.image_rgb.shape)
            with open(self.label_out_path.as_posix() + '/' + os.path.splitext(img_name)[0] + '.txt', "w") as f:
                for label in labels:
                    f.write(label)
                    f.write('\n')
            print("Label output path: {}".format(self.label_out_path))
            self.dump_yaml(self.data_output_path.as_posix())
            print("******")
            return

    def check_id(self, rgb_img_path, seg_img_path):
        img_name = os.path.splitext(os.path.basename(rgb_img_path))[0]
        seg_name = os.path.splitext(os.path.basename(seg_img_path))[0]
        if img_name != seg_name:
            print("Img name error: {} {}".format(img_name, seg_name))
            return False
        else:
            return True

    def dump_yaml(self, dataset_path):
        coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                      'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                      'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                      'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                      'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear',
                      'hair drier', 'toothbrush', 'traffic light red', 'traffic light yellow', 'traffic light green']
        dict_file = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/train',
            'test': '',
            'nc': len(coco_names),
            'names': coco_names
        }
        with open(dataset_path + '/../yolo_coco_carla.yaml', 'w') as file:
            yaml.dump(dict_file, file)
        # with open(dataset_path + '/yolo_coco_carla.yaml', 'r') as file:
        #     t = yaml.safe_load(file)
        #     print(t)

    def decrease_brightness(self, img, value=30):
        h, s, v = cv2.split(img)
        lim = 0 + value
        v[v < lim] = lim
        v[v >= lim] -= lim
        final_hsv = cv2.merge((h, s, v))
        return final_hsv

    def check_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = self.decrease_brightness(hsv, 80)

        red_min = np.array([0, 5, 150])
        red_max = np.array([10, 255, 255])
        red_min_2 = np.array([175, 5, 150])
        red_max_2 = np.array([180, 255, 255])

        yellow_min = np.array([25, 5, 150])
        yellow_max = np.array([35, 180, 255])

        green_min = np.array([35, 5, 150])
        green_max = np.array([90, 255, 255])

        red_thresh = cv2.inRange(hsv_img, red_min, red_max) + cv2.inRange(hsv_img, red_min_2, red_max_2)
        yellow_thresh = cv2.inRange(hsv_img, yellow_min, yellow_max)
        green_thresh = cv2.inRange(hsv_img, green_min, green_max)

        red_blur = cv2.medianBlur(red_thresh, 5)
        yellow_blur = cv2.medianBlur(yellow_thresh, 5)
        green_blur = cv2.medianBlur(green_thresh, 5)

        red = cv2.countNonZero(red_blur)
        yellow = cv2.countNonZero(yellow_blur)
        green = cv2.countNonZero(green_blur)

        light_color = max(red, green, yellow)
        if light_color > self.color_pixels_min:
            if light_color == red:
                return LABEL_ID.TRAFFIC_LIGHT_RED
            elif light_color == yellow:
                if self.world_name == 'Carla/Maps/Town10HD_Opt':
                    return LABEL_ID.TRAFFIC_LIGHT
                return LABEL_ID.TRAFFIC_LIGHT_YELLOW
            elif light_color == green:
                return LABEL_ID.TRAFFIC_LIGHT_GREEN
        else:
            return LABEL_ID.TRAFFIC_LIGHT


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record_id',
        default='record2021_1106_0049',
        help='record_id of raw data')
    args = argparser.parse_args()
    data_path = RAW_DATA_PATH / Path(args.record_id)
    print(data_path)
    vehicle_data_list = glob.glob(data_path.as_posix() + '/vehicle*' + '/vehicle*')
    for vehicle_data_path in vehicle_data_list:
        yolo_label_manager = YoloLabel(vehicle_data_path)
        yolo_label_manager.process()
    return


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)
