# run_sensorfusion_detect.py
import os, subprocess, argparse
from to_kitti_format import save_pred_dir_to_kitti

def run_sensorfusion_detection(kroot, repo, preds_dir, img_size):
    labels = os.path.join(preds_dir,'labels')
    os.makedirs(labels, exist_ok=True)
    # 1) LidarObstacleDetection
    subprocess.run([
      'python3', os.path.join(repo,'LidarObstacleDetection','detect_lidar.py'),
      '--pcd_path',   f"{kroot}/velodyne",
      '--calib_path', f"{kroot}/calib",
      '--out_labels', labels
    ], check=True)
    # 2) 2D_Feature_Tracking
    subprocess.run([
      'python3', os.path.join(repo,'2D_Feature_Tracking','track_features.py'),
      '--image_path',  f"{kroot}/image_02",
      '--labels_path', labels,
      '--out_labels',  labels
    ], check=True)
    save_pred_dir_to_kitti(labels, args.kitti_out_dir, img_size)
    print("SensorFusion results →", args.kitti_out_dir)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kitti_root',   required=True)
    p.add_argument('--repo_root',    required=True)
    p.add_argument('--preds_dir',    required=True)
    p.add_argument('--kitti_out_dir',required=True)
    p.add_argument('--img_w', type=int, default=1242)
    p.add_argument('--img_h', type=int, default=375)
    args=p.parse_args()
    run_sensorfusion_detection(args.kitti_root, args.repo_root, args.preds_dir,
                               (args.img_w, args.img_h))
