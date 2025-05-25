# run_ocsort.py
# OC-SORT 트래킹 래퍼 (로컬 클론된 ocsort_local 사용)

import os
import sys
import glob
import argparse
import numpy as np

# 스크립트 위치 기준으로 ocsort_local 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ocsort_local = os.path.join(BASE_DIR, 'repos', 'ocsort_local')
if not os.path.isdir(ocsort_local):
    raise FileNotFoundError(f"ocsort_local 폴더를 찾을 수 없습니다: {ocsort_local}")
if ocsort_local not in sys.path:
    sys.path.insert(0, ocsort_local)

try:
    # 경로: repos/ocsort_local/trackers/ocsort_tracker/ocsort.py
    from trackers.ocsort_tracker.ocsort import OCSort
except Exception as e:
    raise ImportError(f"OCSort 모듈 로드 실패: {e}")

def run_ocsort(det_dir, track_out_dir, img_size=(1242,375)):
    """
    OC-SORT 트래킹 실행
    det_dir: 검출 결과 txt 폴더 (KITTI 포맷)
    track_out_dir: 트래킹 결과 저장 폴더
    img_size: (width, height) 튜플
    """
    os.makedirs(track_out_dir, exist_ok=True)
    tracker = OCSort(det_thresh=0.3)  # 검출 임계값

    # img_info: (height, width)
    img_info = (img_size[1], img_size[0])

    det_files = sorted(glob.glob(os.path.join(det_dir, '*.txt')))
    for det_path in det_files:
        frame_id = int(os.path.splitext(os.path.basename(det_path))[0])
        dets = []
        for line in open(det_path):
            vals = line.strip().split()
            if len(vals) < 6:
                continue
            cls, x1, y1, x2, y2, score = map(float, vals)
            dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets)

        if dets.size == 0:
            tracks = []
        else:
            tracks = tracker.update(dets, img_info, img_size)

        # 트랙 결과를 파일 단위로 작성
        out_file = os.path.join(track_out_dir, f"{frame_id:06d}.txt")
        with open(out_file, 'w') as f:
            for t in tracks:
                # t may have 5 or 6 elements
                if len(t) == 6:
                    x1, y1, x2, y2, tid, sc = t
                elif len(t) == 5:
                    x1, y1, x2, y2, tid = t
                    sc = 1.0
                else:
                    continue
                line = f"{frame_id} {int(tid)} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {sc:.4f}\n"
                f.write(line)

    print("완료: OC-SORT 트래킹 →", track_out_dir)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--det_dir',       required=True, help='Detection txt 폴더')
    p.add_argument('--track_out_dir', required=True, help='Tracking 결과 폴더')
    p.add_argument('--img_size',      nargs=2, type=int, default=[1242,375], help='이미지 크기 W H')
    args = p.parse_args()
    run_ocsort(args.det_dir, args.track_out_dir, tuple(args.img_size))
