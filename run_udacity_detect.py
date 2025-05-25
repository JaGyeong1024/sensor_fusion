# run_udacity_detect.py
import os, subprocess, argparse

def run_udacity_detection(kitti_root, repo_root, preds_dir):
    os.makedirs(preds_dir, exist_ok=True)
    exe = os.path.join(repo_root, 'build', '3D_object_tracking')
    cmd = [
        exe,
        'detector',
        'FAST', 'ORB',                                  # detectorType, descriptorType
        os.path.join(kitti_root, 'image_02'),
        os.path.join(kitti_root, 'calib'),
        preds_dir
    ]
    print("실행:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--repo_root',  required=True)
    p.add_argument('--preds_dir',  required=True)
    args = p.parse_args()

    run_udacity_detection(
        args.kitti_root,
        args.repo_root,
        args.preds_dir
    )
    print("Udacity Detector 결과 →", args.preds_dir)
