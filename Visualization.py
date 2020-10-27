import os
import sys
import math
import random
import time
import numpy as np
import open3d as o3d
from utils.calibration import Calibration
from utils.testo3d import get_strong
def get_fov_flag(pts_rect, img_shape, calib):
    '''
    Valid point should be in the image (and in the PC_AREA_SCOPE)
    :param pts_rect:
    :param img_shape:
    :return:
    '''
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    # print(pts_rect_depth)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[0])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[1])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    pts_valid_flag = np.logical_and(pts_valid_flag, pts_rect_depth <= 1000)
    return pts_valid_flag

def custom_draw_geometry(vis, geometry_list, map_file=None, recording=False,param_file='test.json'):
    vis.clear_geometries()
    paper = True
    
    if paper:
        geometry_list = get_strong(geometry_list)
    for pcd in geometry_list:
        vis.add_geometry(pcd)
    param = o3d.io.read_pinhole_camera_parameters(param_file)

    ctr = vis.get_view_control()
    # ctr.set_zoom(0.4)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    # time.sleep(5)
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('test.json', param)
    if recording:
        vis.capture_screen_image(map_file,True)
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(param_file, param)

def get_global_bboxes(global_labels):
    global_bboxes = []
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, np.pi])
    for _label in global_labels:
        if 'sensor' in _label[0]:
            continue
        bbox_center = np.array(
            list(map(float, _label[2:5]))) + np.array([0, 0, float(_label[10])])
        bbox_center[1] *= -1
        # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(list(map(np.radians,map(float, _label[5:8]))))
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(
            np.array([0, 0, -np.radians(float(_label[7]))]))
        bbox_extend = np.array([float(num)*2 for num in _label[8:11]])
        global_bbox = o3d.geometry.OrientedBoundingBox(
            bbox_center, bbox_R, bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(
            global_bbox)
        # lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
        lineset.paint_uniform_color([1.,0.,0.])
        # lineset.rotate(R,center=[0,0,0])
        global_bboxes.append(lineset)
        # print('-----------')
    return global_bboxes

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

    # return [line[:-1] for line in lines]
    return ["{:010d}.txt".format(int(line)) for line in lines]

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
    # if path:
    #     if not os.path.exists(path[:-14]):
    #         os.makedirs(path[:-14])
    #     np.savetxt(path, np.array(labels), fmt='%s', delimiter=' ')
    return bboxes

# from visualization_labels import get_fov_flag
def get_pcd(pointcloud, calib, img_shape=[1242,375]):
    pcd = o3d.geometry.PointCloud()
    pts_rect = calib.lidar_to_rect(pointcloud[:, :3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    # points = pointcloud[fov_flag][:,:3]
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:,:3])
    # pcd.paint_uniform_color([1,1,1])
    pcd.paint_uniform_color([0,0,0])
    return [pcd]

def custom_draw_geometry(vis, geometry_list, map_file=None, recording=False,param='nb.json'):
    vis.clear_geometries()
    for geometry in geometry_list:
        R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0,0,np.pi/2]))
        geometry.rotate(R,center=[0,0,0])
        paper = True
        
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
    # print(len(sys.argv))
    tmp_car = []
    tmp_frame = []
    if len(sys.argv) == 4:
        tmp_car.append(str(sys.argv[2]))
        tmp_frame.append(str(sys.argv[3]))

    visualization_3d = True
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=540*2, height=540*2, left=5, top=5)
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.get_render_option().show_coordinate_frame = False
    vis.get_render_option().point_size = 2
    vis.get_render_option().line_width = 10.0
    # vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.YCoordinate
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)

    root_path = sys.argv[1]
    gt_split_file = root_path + '/img_list.txt'
    figure_root = root_path + '/video/federated/'
    # if not os.path.exists(figure_root):
    #     os.makedirs(figure_root)
    vehicle_list = [v for v in os.listdir(root_path) if 'vehicle' in v]
    frame_id_list = _read_imageset_file(gt_split_file)
    for vehicle_id in vehicle_list:
        for frame_id in frame_id_list:
            if vehicle_id[-3:] not in tmp_car or str(int(frame_id[:-4])) not in tmp_frame:
                if len(tmp_car) == 0 and len(tmp_frame) == 0:
                    pass
                else:
                    continue
            # print(frame_id)
            # if not os.path.exists(figure_root+vehicle_id):
            #     os.makedirs(figure_root+vehicle_id)
            figure_file = figure_root + vehicle_id + '/' + frame_id[:-3] + 'png'
            gt_file = root_path + '/' + vehicle_id + '/label00/' + frame_id
            pcd_file = root_path + '/' + vehicle_id + '/velodyne/' + frame_id[:-3] + 'bin'
            calib_file = root_path + '/' + vehicle_id + '/calib00/' + frame_id
            pretrain_file = root_path + '/' + vehicle_id + '/federated_test/' + frame_id

            pretrain_file_ = root_path[:-1] + '/' + vehicle_id + '/federated_test_fusion/' + frame_id
            # pretrain_file = root_path + '/' + vehicle_id + '/federated_test/' + frame_id
            pretrain_fusion = root_path + '/dynamic/pretrain_test_fusion/' + vehicle_id + '/' + frame_id
            pretrain_fusion_file = root_path + '/' + vehicle_id + '/federated_test/' + frame_id 
            calib = Calibration(calib_file)
            pcd = get_pcd(np.fromfile(pcd_file, dtype=np.dtype('f4'), count=-1).reshape([-1, 4]), calib)
            gt_label,pretrain_label,pretrain_fusion_label = get_label(gt_file),get_label(pretrain_file),get_label(pretrain_fusion_file)
            gt_bbox = get_bboxes(gt_label,calib,[1.,0.,0.])
            pretrain_bbox = get_bboxes(pretrain_label,calib,[0.4,0.4,0.4],False,pcd,pretrain_file_)
            # print(pretrain_file)
            pretrain_fusion_bbox = get_bboxes(pretrain_fusion_label,calib,[0.7,0.7,0.7])
            # print(pretrain_fusion_file)
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
            # geometry_list = [mesh] + pcd + gt_bbox + pretrain_fusion_bbox# + pretrain_fusion_bbox
            geometry_list = pcd + gt_bbox + pretrain_bbox + pretrain_fusion_bbox + [mesh]
            # save_view_point(pcd[0],'nb.json')
            print(figure_file)
            custom_draw_geometry(vis,geometry_list,figure_file,False)
