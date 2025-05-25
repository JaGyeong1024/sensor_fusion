# run_sort.py
# SORT 트래킹 스크립트 수정 버전
import os
import glob
import argparse
import numpy as np
import sys

# 로컬 sort 모듈 로드 (UROP/sort/sort.py)
sort_dir = os.path.join(os.path.dirname(__file__), 'sort')
if sort_dir not in sys.path:
    sys.path.insert(0, sort_dir)
try:
    from sort import Sort
except ImportError:
    raise ImportError(f"Cannot import Sort from {sort_dir}/sort.py")


def run_sort(det_dir, track_out_dir, max_age=30, min_hits=3, iou_threshold=0.3):
    os.makedirs(track_out_dir, exist_ok=True)
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    # detection files: frame_id.txt containing 'cls x1 y1 x2 y2 score'
    det_files = sorted(glob.glob(os.path.join(det_dir, '*.txt')))
    for det_path in det_files:
        frame_id = int(os.path.splitext(os.path.basename(det_path))[0])
        # load detections
        dets = []
        for line in open(det_path):
            vals = line.strip().split()
            if len(vals) < 6:
                continue
            cls, x1, y1, x2, y2, score = map(float, vals)
            dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets)
        # update tracker
        
        # update tracker
        if dets.size == 0:
            # no detections: advance tracker, but no output tracks
            tracks = np.empty((0,5))
        else:
            tracks = tracker.update(dets)

        # save tracks: frame_id, track_id, bbox, score
        out_file = os.path.join(track_out_dir, f"{frame_id:06d}.txt")
        with open(out_file, 'w') as f:
            for track in tracks:
                tid, x1, y1, x2, y2 = track.astype(int)
                # SORT에는 score 없어, append dummy score
                f.write(f"{frame_id} {tid} {x1} {y1} {x2} {y2} 1.0\n")
    print("완료: SORT 트래킹 →", track_out_dir)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--det_dir',       required=True, help='Detection txt 폴더')
    p.add_argument('--track_out_dir', required=True, help='Tracking 결과 폴더')
    args = p.parse_args()
    run_sort(args.det_dir, args.track_out_dir)