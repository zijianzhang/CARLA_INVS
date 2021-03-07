import os
import sys
import math
import random
import time
import numpy as np
import open3d as o3d
from text_3d import text_3d
from calibration import Calibration
from Fusion import get_global_bboxes, custom_draw_geometry

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return ["{:010d}.txt".format(int(line)) for line in lines]

def get_bboxes(labels, color=[0.5,0.5,0.5],tmp=0.01):
    bboxes = []
    for label in labels:
        label = [float(i) for i in label]
        label[2] += tmp
        label[1] += tmp
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,label[6]])
        bbox = o3d.geometry.OrientedBoundingBox(label[:3],R,label[3:6])
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color(color)  
        bboxes += [lineset]
    return bboxes

if __name__ == "__main__":
    visualization_3d = True
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960*2, height=640*2, left=5, top=5)
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    vis.get_render_option().show_coordinate_frame = False
    vis.get_render_option().point_size = 1
    vis.get_render_option().line_width = 10.0
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)

    root_path = sys.argv[1] if len(sys.argv>1) else ''
    gt_split_file = root_path + '/img_list_test.txt'
    figure_root = root_path + '/video/global/federated/'
    if not os.path.exists(figure_root):
        os.makedirs(figure_root)
    vehicle_list = [v for v in os.listdir(root_path) if 'vehicle' in v]
    frame_id_list = _read_imageset_file(gt_split_file)
    global_gt_path = root_path + '/global_label'
    for frame_id in frame_id_list:
        # if '7944' not in frame_id:
        #     continue
        geometry_list = []
        figure_file = figure_root + '/' + frame_id[:-3] + 'png'
        frame_global_label_path = global_gt_path + '/' + frame_id
        frame_label = np.loadtxt(frame_global_label_path, dtype='str', delimiter=' ')
        global_bboxes = get_global_bboxes(frame_label)
        geometry_list += global_bboxes
        pretrain_file = root_path + '/dynamic/pretrain_test_fusion/global/'+frame_id
        distill_file = root_path + '/dynamic/distill_test_fusion/global/'+frame_id
        federated_file = root_path + '/dynamic/federated_test_fusion/global/'+frame_id
        pretrain_label = np.loadtxt(pretrain_file, dtype='str', delimiter=' ')
        distill_label = np.loadtxt(distill_file, dtype='str', delimiter=' ')
        federated_label = np.loadtxt(federated_file, dtype='str', delimiter=' ')
        pretrain_bboxes = get_bboxes(pretrain_label,[0.9,0.1,0.1])
        geometry_list += pretrain_bboxes
        distill_bboxes = get_bboxes(distill_label,[0.1,0.9,0.1])
        geometry_list += distill_bboxes
        federated_bboxes = get_bboxes(federated_label,[0.1,0.1,0.9])
        geometry_list += federated_bboxes
        print(figure_file)

        custom_draw_geometry(vis, geometry_list, figure_file, False,'global.json')
        pass
