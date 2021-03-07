# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)

        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def get_top_rect(corners):
    rect = [(corners[0,0], corners[0,1])] + [(corners[3,0], corners[3,1])] + [(corners[4,0], corners[4,1])] + [(corners[7,0], corners[7,1])]
    return rect

def box3d_iou(corners1, corners2, vis=True):
    corners1 = get_3d_box(corners1)
    corners2 = get_3d_box(corners2)
    
    bbox1 = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(corners1)))
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox1)
    lineset.paint_uniform_color(np.array([0.9, 0.1, 0.1]))
    bbox2 = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(corners2)))
    # if vis:
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    # rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    # rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    rect1 = [(corners1[i,0], corners1[i,1]) for i in [0,3,7,4]]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in [0,3,7,4]]
    # rect1 = get_top_rect(corners1)
    # rect2 = get_top_rect(corners2)
    # print('rect1',rect1)
    # print('rect2',rect2)
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    # print(area1,area2)
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    # print(inter_area)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[4,2], corners2[4,2])
    ymin = max(corners1[5,2], corners2[5,2])
    # print(ymax,ymin)
    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    # print(inter_vol, (vol1 + vol2 - inter_vol))
    # if round(iou_2d,2) > 0:
    #     # print(iou,vol1,vol2,inter_vol,inter_area)
    #     # print(iou_2d,inter_area,area1,area2)
    #     # print(inter_area,ymax,ymin)
    # o3d.visualization.draw_geometries([lineset,bbox2,mesh])
    # print(iou,iou_2d)
    return round(iou,2), round(iou_2d,2)

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(bbox):#, box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,-s,0],
                        [s,c,0],
                        [0,0,1]])
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(bbox[6])
    # print(R)
    R1 = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,bbox[6]])
    # print(R1)
    l,h,w = bbox[3],bbox[4],bbox[5]
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox[0]# +100;
    corners_3d[1,:] = corners_3d[1,:] + bbox[1]# +100;
    corners_3d[2,:] = corners_3d[2,:] + bbox[2]# +100;
    corners_3d = np.transpose(corners_3d)
    # import open3d as o3d 
    tmp_R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,bbox[6]])
    tmp_bbox = o3d.geometry.OrientedBoundingBox(bbox[:3],tmp_R,bbox[3:6])
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(tmp_bbox)
    lineset.paint_uniform_color(np.array([0.1, 0.1, 0.9]))
    # print('---------------')
    # # print(np.asarray(tmp_bbox.get_box_points(),dtype='float32'))
    # print(corners_3d)
    # print('---------------')
    # return np.asarray(tmp_bbox.get_box_points(),dtype='float32')
    return corners_3d#,lineset
    
if __name__=='__main__':
    print('------------------')
    # get_3d_box(box_size, heading_angle, center)
    corners_3d_ground  = get_3d_box((1.497255,1.644981, 3.628938), -1.531692, (2.882992 ,1.698800 ,20.785644)) 
    corners_3d_predict = get_3d_box((1.458242, 1.604773, 3.707947), -1.549553, (2.756923, 1.661275, 20.943280 ))
    (IOU_3d,IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)
    print (IOU_3d,IOU_2d) #3d IoU/ 2d IoU of BEV(bird eye's view)
      
