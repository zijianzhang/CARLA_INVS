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



def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    print(vis)
    vis.create_window()
    vis.create_window(width=540*2, height=540*2, left=5, top=5)
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().show_coordinate_frame = False
    vis.get_render_option().point_size = 3
    vis.add_geometry(pcd)
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

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return ["{:010d}.txt".format(int(line[:-5])) for line in lines]

def get_label(label_path):
    try:
        try:
            label = np.loadtxt(label_path, dtype='str',
                                delimiter=' ').reshape((-1, 16))
        except:
            label = np.loadtxt(label_path, dtype='str',
                           delimiter=' ').reshape((-1, 15))
    except OSError:
        label = []
    return label

def get_bboxes(labels, calib, color=[0.5,0.5,0.5], pre=False,pcd=None,path=None):
    bboxes = []
    for label in labels:
        if not pre:
            if label[0] != 'Car' or float(label[-1]) < -1.5 :
                continue
        else:

            if label[0] != 'Car' or float(label[-1]) < -1 :
                continue
        bbox_center = np.array(list(map(float,label[11:14]))).reshape(1,3) - np.array([0,float(label[8]),0]) / 2
        distance = float(label[13])
        bbox_center = calib.rect_to_lidar(bbox_center).flatten()
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz  (np.array([0,0,-np.pi/2-float(label[14])]))
        bbox_extend = np.array(list(map(float,label[8:11]))[::-1]) 
        bbox = o3d.geometry.OrientedBoundingBox(bbox_center,bbox_R,bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color(np.array(color))
        bboxes.append(lineset)
        if pcd:
            tmp2 = pcd[0].crop(bbox)
            size = 0
            print(distance)
            if len(tmp2.points) + distance < 250:
                size = 1
            if len(tmp2.points) + distance < 125:
                size = 2
            label[2] = size
    if path:
        if not os.path.exists(path[:-14]):
            os.makedirs(path[:-14])
        np.savetxt(path, np.array(labels), fmt='%s', delimiter=' ')
    return bboxes

from visualization_labels import get_fov_flag
def get_pcd(pointcloud, calib, img_shape=[1242,375]):
    pcd = o3d.geometry.PointCloud()
    pts_rect = calib.lidar_to_rect(pointcloud[:, :3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    # points = pointcloud[fov_flag][:,:3]
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
    # pcd.paint_uniform_color([1,1,1])
    # pcd.paint_uniform_color([0,0,0])
    return [pcd]

def custom_draw_geometry(vis, geometry_list, map_file=None, recording=False,param='nb.json'):
    vis.clear_geometries()
    for geometry in geometry_list:
        R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0,0,np.pi/2]))
        geometry.rotate(R,center=[0,0,0])

        paper = True
        from testo3d import get_strong
        if paper:
            geometry = get_strong([geometry])
            for g in geometry:
                vis.add_geometry(g)
        else:
            vis.add_geometry(geometry)
    param = o3d.io.read_pinhole_camera_parameters(param)
    ctr = vis.get_view_control()
    # ctr.set_zoom(0.4)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('test.json', param)
    if recording:
        vis.capture_screen_image(map_file,True)

if __name__ == "__main__":
    color = [
        [0.1, 0.9, 0.9],
        [0.1, 0.9, 0.1],
        [0.9, 0.1, 0.1],
        [0.9, 0.9, 0.1],
        [0.9, 0.1, 0.9],
        [0.9, 0.9, 0.9],
    ]

    visualization_3d = True
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=540*2, height=540*2, left=5, top=5)
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    vis.get_render_option().show_coordinate_frame = False
    vis.get_render_option().point_size = 2
    vis.get_render_option().line_width = 10.0
    # # vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.YCoordinate
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)

    root_path = sys.argv[1]
    gt_split_file = root_path + '/img_list.txt'
    figure_root = root_path + '/video/federated/'
    if not os.path.exists(figure_root):
        os.makedirs(figure_root)
    vehicle_list = [v for v in os.listdir(root_path) if 'vehicle' in v]
    frame_id_list = _read_imageset_file(gt_split_file)

    tmp_car = ['231','237','233','247']
    for vehicle_id in vehicle_list:
        for frame_id in frame_id_list:
            # if '8449' not in frame_id or '595' not in vehicle_id:
            #     continue
            # print(frame_id)
            if not os.path.exists(figure_root+vehicle_id):
                os.makedirs(figure_root+vehicle_id)
            figure_file = figure_root + vehicle_id + '/' + frame_id[:-3] + 'png'
            gt_file = root_path + '/' + vehicle_id + '/label00/' + frame_id
            pcd_file = root_path + '/' + vehicle_id + '/velodyne/' + frame_id[:-3] + 'bin'
            calib_file = root_path + '/' + vehicle_id + '/calib00/' + frame_id
            # # pretrain_file = root_path + '/' + vehicle_id + '/federated_test/' + frame_id

            # pretrain_file_ = root_path[:-1] + '/' + vehicle_id + '/federated_test_fusion/' + frame_id
            # # pretrain_file = root_path + '/' + vehicle_id + '/federated_test/' + frame_id
            # pretrain_fusion = root_path + '/dynamic/pretrain_test_fusion/' + vehicle_id + '/' + frame_id
            # pretrain_fusion_file = root_path + '/' + vehicle_id + '/federated_test/' + frame_id 
            calib = Calibration(calib_file)
            pcd = get_pcd(np.fromfile(pcd_file, dtype=np.dtype('f4'), count=-1).reshape([-1, 4]), calib)
            # gt_label,pretrain_label,pretrain_fusion_label = get_label(gt_file),get_label(pretrain_file),get_label(pretrain_fusion_file)
            gt_label = get_label(gt_file)
            print(len(gt_label))
            gt_bbox = get_bboxes(gt_label,calib,[1.,1.,1.])
            # pretrain_bbox = get_bboxes(pretrain_label,calib,[0.4,0.4,0.4],True,pcd,pretrain_file_)
            # # print(pretrain_file)
            # pretrain_fusion_bbox = get_bboxes(pretrain_fusion_label,calib,[0.7,0.7,0.7])
            # # print(pretrain_fusion_file)
            # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
            # geometry_list = [mesh] + pcd + gt_bbox + pretrain_fusion_bbox# + pretrain_fusion_bbox
            geometry_list = pcd + gt_bbox# + pretrain_bbox + pretrain_fusion_bbox
            # save_view_point(pcd[0],'nb.json')
            # print(figure_file)
            custom_draw_geometry(vis,geometry_list,figure_file,True)
            # exit()
        #         ego_point_cloud, color[index], ego_location, fusion_rotation, location[index], calib, fusion_location)
        #     ego_gt_bboxes,_ = get_ego_bboxes(get_ego_data(
        #         ego_label_file), ego_location, ego_rotation, calib, image_location=location[index],fusion=fusion,ROI=ROI,fusion_location=fusion_location,fusion_rotation=fusion_rotation,ego_vehicle_data=ego_vehicle_data)
        #     ego_bboxes,_ = get_ego_bboxes(get_ego_data(
        #         ego_data_file), ego_location, ego_rotation, calib, color=color[index], image_location=location[index],fusion=fusion,ROI=ROI,fusion_location=fusion_location,fusion_rotation=fusion_rotation,ego_vehicle_data=ego_vehicle_data)
        #     # print(len(ego_gt_bboxes))
        #     geometry_list += ego_bboxes + ego_gt_bboxes
        # # image_path = 
        # custom_draw_geometry(vis, geometry_list, map_file, True,'test.json')
        # exit()
