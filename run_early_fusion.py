#run_early_fusion.py
# Early Fusion: YOLOv5n + LiDAR 포인트 융합 검출 (이미지+LiDAR)
import os
import subprocess
import argparse

def run_early_fusion(kroot, repos_root, preds_dir, model_path, calib_file, img_size):
    os.makedirs(preds_dir, exist_ok=True)
    cmd = [
        'python3', os.path.join(repos_root, 'Camera-Lidar-Sensor-Fusion', 'early_fusion.py'),
        '--model',      model_path,
        '--img_path',   os.path.join(kroot, 'image_02'),
        '--pcd_path',   os.path.join(kroot, 'velodyne'),
        '--calib_file', calib_file,
        '--save_txt',   preds_dir,
        '--img_size',   str(img_size[0]), str(img_size[1])
    ]
    print("실행:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--repos_root', required=True)
    p.add_argument('--preds_dir',   required=True)
    p.add_argument('--model_path',  required=True)
    p.add_argument('--calib_file',  required=True)
    p.add_argument('--img_size',    nargs=2, type=int, default=[1242,375])
    args = p.parse_args()
    run_early_fusion(
        args.kitti_root,
        args.repos_root,
        args.preds_dir,
        args.model_path,
        args.calib_file,
        args.img_size
    )
    print(f"완료: {args.preds_dir}")