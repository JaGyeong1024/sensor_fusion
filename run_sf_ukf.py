# run_sf_ukf.py

import os
import subprocess
import argparse

def run_ukf(kitti_root, repo_root, det_dir, track_out_dir):
    """
    LiDARObstacleDetection 검출 결과(det_dir)와 KITTI calib를 이용해
    UKF 트래킹을 실행하고 프레임별 결과를 track_out_dir에 저장합니다.
    """
    os.makedirs(track_out_dir, exist_ok=True)

    # UKF 이진 파일 자동 탐색
    ukf_build_dir = os.path.join(repo_root, 'UKF', 'build')
    if not os.path.isdir(ukf_build_dir):
        raise FileNotFoundError(f"UKF build 폴더가 없습니다: {ukf_build_dir}")

    # 실행 가능한 파일 중 첫 번째를 사용
    candidates = []
    for f in os.listdir(ukf_build_dir):
        fp = os.path.join(ukf_build_dir, f)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            candidates.append(fp)
    if not candidates:
        raise FileNotFoundError(f"UKF 빌드 폴더에 실행 파일이 없습니다: {ukf_build_dir}")
    ukf_exe = candidates[0]

    # 명령어 구성
    cmd = [
        ukf_exe,
        '--det_path', det_dir,
        '--calib',     os.path.join(kitti_root, 'calib', '000000.txt'),
        '--out_dir',   track_out_dir
    ]
    print("실행:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_root',    required=True, help='training 폴더 경로')
    parser.add_argument('--repo_root',     required=True, help='SensorFusion 레포 경로')
    parser.add_argument('--det_dir',       required=True, help='검출 결과 폴더 (KITTI 포맷)')
    parser.add_argument('--track_out_dir', required=True, help='출력 트래킹 결과 폴더')
    args = parser.parse_args()

    run_ukf(
        args.kitti_root,
        args.repo_root,
        args.det_dir,
        args.track_out_dir
    )
    print(f"UKF 트래킹 결과가 {args.track_out_dir}에 저장되었습니다.")
