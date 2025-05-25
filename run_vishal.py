# run_vishal.py

"""
run_vishal.py
early_fusion.py 를 호출해 KITTI 이미지 기준으로 PCD가 있는 프레임만 처리합니다.
"""

import os
import subprocess
import argparse

def run_early_fusion(kroot, out_dir, model, calib_file, img_size):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        'python3', 'repos/Camera-Lidar-Sensor-Fusion/early_fusion.py',
        '--model',      model,
        '--img_path',   os.path.join(kroot, 'image_02'),
        '--pcd_path',   os.path.join(kroot, 'velodyne'),
        '--calib_file', calib_file,
        '--save_txt',   out_dir,
        '--img_size',   str(img_size[0]), str(img_size[1])
    ]
    print("실행:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_root', required=True, help='training 폴더 경로')
    parser.add_argument('--preds_dir',   required=True, help='출력 txt 폴더 (preds_vishal)')
    parser.add_argument('--model_path',  required=True, help='YOLOv5n best.pt 경로')
    parser.add_argument('--calib_file',  required=True, help='단일 calib txt 파일 경로')
    parser.add_argument('--img_size',    nargs=2, type=int, default=[1242,375])
    args = parser.parse_args()

    run_early_fusion(
        args.kitti_root,
        args.preds_dir,
        args.model_path,
        args.calib_file,
        tuple(args.img_size)
    )
    print(f"완료: {args.preds_dir}")
