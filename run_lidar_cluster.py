# run_lidar_cluster.py
# LiDAR 포인트 클러스터링 → 2D 바운딩박스 생성
import os
import glob
import argparse
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def parse_calib(filepath):
    """KITTI calib 파일 파싱: 'key: values' 또는 'key values' 형태 지원"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # key: val or key val
            if ':' in line:
                key, val = line.split(':',1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key, val = parts[0], ' '.join(parts[1:])
            try:
                arr = np.array([float(x) for x in val.split()])
                data[key] = arr
            except Exception:
                continue
    return data


def run_lidar_cluster(kroot, preds_dir, calib_file, eps=0.5, min_samples=10):
    os.makedirs(preds_dir, exist_ok=True)
    calib = parse_calib(calib_file)
    print("Calibration keys:", calib.keys())
    # Transform matrix key detection
    if 'Tr_velo_cam' in calib:
        Tr = calib['Tr_velo_cam']
    elif 'Tr_velo_to_cam' in calib:
        Tr = calib['Tr_velo_to_cam']
    else:
        raise KeyError("No 'Tr_velo_cam' or 'Tr_velo_to_cam' found in calib file")
    R0 = calib.get('R_rect', calib.get('R0_rect'))
    P2 = calib.get('P2', calib.get('P2_rect'))
    if Tr.size != 12 or R0.size not in (9,) or P2.size not in (12,):
        raise ValueError("Calibration arrays have incorrect sizes: Tr={}, R0={}, P2={}".format(Tr.size, R0.size, P2.size))
    Tr = Tr.reshape(3,4)
    R0 = R0.reshape(3,3)
    P2 = P2.reshape(3,4)

    pcd_files = sorted(glob.glob(os.path.join(kroot, 'velodyne', '*.pcd')))
    if not pcd_files:
        print("경고: velodyne 폴더에 pcd 파일이 없습니다.")
    for vf in pcd_files:
        fid = os.path.splitext(os.path.basename(vf))[0]
        try:
            pts = np.asarray(o3d.io.read_point_cloud(vf).points)
        except Exception as e:
            print(f"[Warning] frame {fid}: PCD 로드 실패 ({e}), 스킵")
            continue
        if pts.size == 0:
            print(f"[Warning] frame {fid}: 포인트 없음, 스킵")
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        out_file = os.path.join(preds_dir, f"{fid}.txt")
        with open(out_file, 'w') as f:
            for lab in set(labels):
                if lab < 0:
                    continue  # noise
                pc = pts[labels == lab]
                hom = np.hstack([pc, np.ones((pc.shape[0],1))])  # N×4
                cam = hom.dot(Tr.T)       # N×3
                rect = cam.dot(R0.T)      # N×3
                img_hom = np.hstack([rect, np.ones((rect.shape[0],1))]).dot(P2.T)  # N×3
                img2 = img_hom[:,:2] / img_hom[:,2:3]
                x1,y1 = img2.min(axis=0)
                x2,y2 = img2.max(axis=0)
                score = len(pc)
                f.write(f"0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.4f}\n")
    print("완료: LiDAR 클러스터 →", preds_dir)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--preds_dir',  required=True)
    p.add_argument('--calib_file', required=True)
    args = p.parse_args()
    run_lidar_cluster(args.kitti_root, args.preds_dir, args.calib_file)