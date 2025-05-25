# load_kitti.py
import os
import cv2
import numpy as np
import open3d as o3d

def load_image(root_dir, frame_id):
    img_path = os.path.join(root_dir, 'image_02', f'{frame_id:06d}.png')
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img

def load_pointcloud(root_dir, frame_id):
    pcd_path = os.path.join(root_dir, 'velodyne', f'{frame_id:06d}.pcd')
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PointCloud not found: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def parse_calib(root_dir, frame_id):
    calib_path = os.path.join(root_dir, 'calib', f'{frame_id:06d}.txt')
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    calib = {}
    with open(calib_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        key, *vals = line.split()
        key = key.rstrip(':')
        vals = list(map(float, vals))
        if key in ['P0','P1','P2','P3']:
            calib[key] = np.array(vals, dtype=np.float64).reshape(3,4)
        elif key in ['R_rect','R0_rect']:
            R = np.array(vals, dtype=np.float64).reshape(3,3)
            R_ext = np.eye(4); R_ext[:3,:3] = R
            calib['R_rect'] = R_ext
        elif key in ['Tr_velo_cam','Tr_velo_to_cam']:
            T = np.eye(4); T[:3,:] = np.array(vals, dtype=np.float64).reshape(3,4)
            calib['Tr_velo_cam'] = T
        elif key=='Tr_imu_velo':
            T = np.eye(4); T[:3,:] = np.array(vals, dtype=np.float64).reshape(3,4)
            calib['Tr_imu_velo'] = T
    if 'P2' in calib and 'R_rect' in calib:
        calib['P2_rect'] = calib['P2'] @ calib['R_rect']
    return calib

def load_kitti(root_dir, frame_id):
    img = load_image(root_dir, frame_id)
    pts = load_pointcloud(root_dir, frame_id)
    calib = parse_calib(root_dir, frame_id)
    return img, pts, calib

if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--frame',    type=int, required=True)
    args = p.parse_args()
    img, pts, calib = load_kitti(args.kitti_root, args.frame)
    print('Image shape:', img.shape)
    print('Pointcloud shape:', pts.shape)
    print('Calib keys:', calib.keys())
