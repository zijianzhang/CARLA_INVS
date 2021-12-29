#!/usr/bin/python3
import argparse
import time

import cv2
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from multiprocessing.dummy import Pool as ThreadPool

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())  # repo path
sys.path.append(Path(__file__).resolve().parent.as_posix())  # file path
from params import *


LABEL_DATAFRAME = pd.DataFrame(columns=['raw_value', 'color', 'coco_names_index'],
                               data=[
                                     # [ 4, (220, 20, 60), 0],
                                     [18, (250, 170, 30), 9],
                                     [12, (220, 220,  0), 80]])

LABEL_COLORS = np.array([
    # (220, 20, 60),   # Pedestrian
    # (0, 0, 142),     # Vehicle
    (220, 220, 0),   # TrafficSign -> COCO INDEX
    (250, 170, 30),  # TrafficLight
])

COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
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
              'hair drier', 'toothbrush', 'traffic sign']


class YoloLabel:
    def __init__(self, data_path, debug=False):
        self.data_path = data_path
        self.data_output_path = Path(data_path) / "../yolo_dataset"
        self.data_output_path = Path(os.path.abspath(self.data_output_path.as_posix()))
        print(self.data_output_path)
        self.image_out_path = self.data_output_path / "images" / "train"
        self.label_out_path = self.data_output_path / "labels" / "train"
        self.image_rgb = None
        self.image_seg = None
        self.preview_img = None
        self.rec_pixels_min = 150
        self.debug = debug
        self.thread_pool = ThreadPool()

    def process(self):
        img_path_list = sorted(glob.glob(self.data_path + '/*.png'))
        img_seg_path_list = sorted(glob.glob(self.data_path + '/seg' + '/*.png'))
        start = time.time()
        self.thread_pool.starmap(self.label_img, zip(img_path_list, img_seg_path_list))
        # for rgb_img, seg_img in zip(img_path_list, img_seg_path_list):
        #     self.label_img(rgb_img, seg_img)
        self.thread_pool.close()
        self.thread_pool.join()
        print("cost: {}s".format(time.time()-start))

    def label_img(self, rgb_img_path, seg_img_path):
        success = self.check_id(rgb_img_path, seg_img_path)
        if not success:
            return
        image_rgb = None
        image_seg = None
        image_rgb = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        image_seg = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
        if image_rgb is None or image_seg is None:
            return
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
        image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGRA2RGB)
        img_name = os.path.basename(rgb_img_path)
        height, width, _ = image_rgb.shape

        labels_all = []
        for index, label_info in LABEL_DATAFRAME.iterrows():
            seg_color = label_info['color']
            coco_id = label_info['coco_names_index']

            mask = (image_seg == seg_color)
            tmp_mask = (mask.sum(axis=2, dtype=np.uint8) == 3)
            mono_img = np.array(tmp_mask * 255, dtype=np.uint8)

            preview_img = image_rgb
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
                max_y, max_x, _ = image_rgb.shape
                if y + h >= max_y or x + w >= max_x:
                    continue

                    # Draw label info to image
                cv2.putText(preview_img, COCO_NAMES[coco_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
                label_info = "{} {} {} {} {}".format(coco_id,
                                                     float(x + (w / 2.0)) / width,
                                                     float(y + (h / 2.0)) / height,
                                                     float(w) / width,
                                                     float(h) / height)
                labels.append(label_info)
                # cv2.imshow("result", self.preview_img)
                # cv2.imshow("test", self.preview_img[y:y+h, x:x+w, :])
                # cv2.waitKey()

            if len(labels) > 0:
                labels_all += labels

        if len(labels_all) > 0:
            os.makedirs(self.label_out_path, exist_ok=True)
            os.makedirs(self.image_out_path, exist_ok=True)
            # print("image\t\t\twidth\theight\n{}\t{}\t{}".format(img_name, width, height))
            # print("Got {} labels".format(len(labels_all)))
            cv2.imwrite(self.image_out_path.as_posix() + '/' + os.path.splitext(img_name)[0] + '.jpg', image_rgb)
            # print(self.image_rgb.shape)
            with open(self.label_out_path.as_posix() + '/' + os.path.splitext(img_name)[0] + '.txt', "w") as f:
                for label in labels_all:
                    f.write(label)
                    f.write('\n')
            # print("Label output path: {}".format(self.label_out_path))
            self.dump_yaml(self.data_output_path.as_posix())
            # print("******")
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
        dict_file = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/train',
            'test': '',
            'nc': len(COCO_NAMES),
            'names': COCO_NAMES
        }
        with open(dataset_path + '/../yolo_coco_carla.yaml', 'w') as file:
            yaml.dump(dict_file, file)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record_id',
        default='record2021_1106_0049',
        help='record_id of raw data')

    argparser.add_argument(
        '--debug',
        default=False
    )

    args = argparser.parse_args()
    debug = args.debug

    data_path = RAW_DATA_PATH / Path(args.record_id)
    print(data_path)
    vehicle_data_list = glob.glob(data_path.as_posix() + '/vehicle*' + '/vehicle*')
    for vehicle_data_path in vehicle_data_list:
        yolo_label_manager = YoloLabel(vehicle_data_path, debug)
        yolo_label_manager.process()
    return


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)
