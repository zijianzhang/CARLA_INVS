import os
import sys
import math
import random
import time
import numpy as np
import open3d as o3d
# from text_3d import text_3d
from calibration import Calibration
from Fusion import get_global_bboxes#, custom_draw_geometry



def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    R = o3d.geometry.get_rotation_matrix_from_xyz([0,np.pi/2,0])
    for p in pcd:
        p.rotate(R,[0,0,0])
        vis.add_geometry(p)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()

def get_scale(global_location,frame_id_list,frame_id,fusion,ego=False):
    ROI = False
    if not fusion:
        location = global_location
        # rotation = global_rotation
    else:
        start = int(frame_id_list[0][:-4])
        end = int(frame_id_list[-1][:-4])
        now = int(frame_id[:-4])
        if now - start < 100:# and now - start < 150:
            ROI = True
            if not ego:
                location = (100-now+start)*np.array(global_location)/100
                # rotation = (100-now+start)*np.array(global_rotation)/100
            else:
                location = (now-start)*np.array(global_location)/100
                # rotation = (now-start)*np.array(global_rotation)/100
        else:
            if not ego:
                location = 0*np.array(global_location)
                # rotation = 0*np.array(global_rotation)
            else:
                location = np.array(global_location)
                # rotation = np.array(global_rotation)
    return location,ROI

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return ["{:010d}.txt".format(int(line)) for line in lines]

if __name__ == "__main__":
    color = [
        [0.9, 0.1, 0.9],
        # [0.5,0.5,0.5],
        [0.1, 0.1, 0.9],
        # [1,1,1],# 
        [0.1, 0.9, 0.1],
        [0.1, 0.9, 0.9],
        [0.9, 0.9, 0.1],
        # [0.9, 0.9, 0.9],
    ]

    global_location = [
        [-300, 0, 0],
        [0, -200, 0],
        [330, 0, 0],
        [-300, -200, 0],
        [300, -200, 0],
        # [200, 200, 0],
        # [200, 0, 0],
        # [200, -200, 0]
    ]
    visualization_3d = True
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960*2, height=640*2, left=5, top=5)
    vis.get_render_option().background_color = np.array([1, 1, 1])
    # vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.get_render_option().show_coordinate_frame = False
    vis.get_render_option().point_size = 2
    vis.get_render_option().line_width = 5.0
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    # vis = vis_init(vis)

    root_path = sys.argv[1]
    frame_view = sys.argv[2]
    fusion = False
    if not fusion:
        map_root = root_path + '/video/'
    else:
        map_root = root_path + '/video/' + '_fusion'

    gt_split_file = root_path + '/img_list.txt'
    vehicle_list = [v for v in os.listdir(root_path) if 'vehicle' in v]
    frame_id_list = _read_imageset_file(gt_split_file)
    global_gt_path = root_path + '/global_label'
    test = 0
    for frame_id in frame_id_list:
        test += 1
        if frame_view not in frame_id:
            continue
        map_file = map_root + '/' + frame_id[:-3] + 'png'
        geometry_list = []

        frame_global_label_path = global_gt_path + '/' + frame_id
        frame_label = np.loadtxt(frame_global_label_path, dtype='str', delimiter=' ')
        global_bboxes = get_global_bboxes(frame_label)
        geometry_list += global_bboxes

        from Visualization_global import get_bboxes

        location,_ = get_scale(global_location,frame_id_list,frame_id,fusion)
        for index,test_id in enumerate(vehicle_list):
            from Fusion import get_ego_file, get_ego_location, get_ego_bboxes, get_ego_data, get_global_pcd
            test_path = root_path + '/' + test_id
            ego_calib_file, ego_label_file, ego_data_file, ego_pointcloud_file = get_ego_file(
                test_path, frame_id, 1)
            calib = Calibration(ego_calib_file)
            ego_point_cloud = np.fromfile(
                ego_pointcloud_file, dtype=np.dtype('f4'), count=-1).reshape([-1, 4])
            ego_location, ego_rotation,ego_vehicle_data = get_ego_location(
                test_id[-3:], frame_label)
            if fusion:
                fusion_location,ROI = get_scale(ego_location,frame_id_list,frame_id,fusion,True)
                fusion_rotation,_ = get_scale(ego_rotation,frame_id_list,frame_id,fusion,True) 
                bias,_ = get_scale([0,0,np.pi/2],frame_id_list,frame_id,fusion,False)
                fusion_rotation += bias
            else:
                fusion_location = np.array([0,0,0])
                fusion_rotation = np.array([0,0,0]) + [0,0,np.pi/2]
                ROI = False
 
            geometry_list += get_global_pcd(
                ego_point_cloud, location=ego_location, sensor_rotation=ego_rotation,color=[0,0,0])

            ego_bboxes,_ = get_ego_bboxes(get_ego_data(
                            ego_data_file), ego_location, ego_rotation, calib, 
                            color=color[index], 
                            image_location=[0,0,0],#location[index],
                            fusion_location=ego_location,##fusion_location,
                            # color=[0.,0.,0.],
                            # image_location=location[index],
                            # fusion_location=fusion_location,
                            fusion=True,
                            ROI=ROI,
                            fusion_rotation=ego_rotation,
                            ego_vehicle_data=ego_vehicle_data)
            
            geometry_list += ego_bboxes # + ego_gt_bboxes

        figure_root = root_path + '/fusion/'
        if not os.path.exists(figure_root):
            os.makedirs(figure_root)
        map_file = figure_root + frame_id[:-3] + 'png'
        print(map_file)
        from Fusion import custom_draw_geometry
        custom_draw_geometry(vis, geometry_list, map_file, True,'test.json',test)


#  python3 Visualization/Visualization_fusion_map.py ./data/record2020_1027_1957 38549