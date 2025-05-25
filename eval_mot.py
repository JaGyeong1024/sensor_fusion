# eval_mot.py
# motmetrics를 이용한 KITTI 2D Tracking 평가 (Car 클래스) - 수정 완료
import os, glob, argparse
import pandas as pd
import motmetrics as mm

# 평가할 메트릭 목록
METRICS = [
    'num_frames', 'mota', 'motp', 'idf1',
    'mostly_tracked', 'mostly_lost',
    'num_false_positives', 'num_misses',
    'num_switches', 'num_fragmentations'
]


def load_kitti_gt(gt_folder):
    """
    KITTI tracking GT (label_02) 폴더에서 Car 클래스만 로드
    motmetrics 포맷(DataFrame) 생성
    """
    rows = []
    files = sorted(glob.glob(os.path.join(gt_folder, '*.txt')))
    for file in files:
        frame = int(os.path.splitext(os.path.basename(file))[0])
        with open(file) as f:
            for line in f:
                vals = line.strip().split()
                track_id = int(vals[0])
                obj_type = vals[1]
                if obj_type != 'Car':
                    continue
                # bbox coords: left, top, right, bottom
                x1 = float(vals[5]); y1 = float(vals[6])
                x2 = float(vals[7]); y2 = float(vals[8])
                width  = x2 - x1
                height = y2 - y1
                rows.append([frame, track_id, x1, y1, width, height])
    df = pd.DataFrame(rows, columns=['FrameId','Id','X','Y','Width','Height'])
    df.set_index(['FrameId','Id'], inplace=True)
    return df


def load_tracker_ts(track_folder):
    """
    Tracker 결과 폴더에서 motmetrics 포맷(DataFrame) 생성
    """
    rows = []
    files = sorted(glob.glob(os.path.join(track_folder, '*.txt')))
    for file in files:
        frame = int(os.path.splitext(os.path.basename(file))[0])
        with open(file) as f:
            for line in f:
                vals = line.strip().split()
                # format: frame, track_id, x1, y1, x2, y2, score
                _, track_id, x1, y1, x2, y2, _ = vals[:7]
                track_id = int(track_id)
                x1 = float(x1); y1 = float(y1)
                x2 = float(x2); y2 = float(y2)
                width  = x2 - x1
                height = y2 - y1
                rows.append([frame, track_id, x1, y1, width, height])
    df = pd.DataFrame(rows, columns=['FrameId','Id','X','Y','Width','Height'])
    df.set_index(['FrameId','Id'], inplace=True)
    return df


def evaluate_tracker(gt_folder, track_folder, iou_threshold=0.5):
    # GT, Tracker DataFrame 생성
    gt = load_kitti_gt(gt_folder)
    ts = load_tracker_ts(track_folder)

    # IoU 기반 매칭
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=iou_threshold)

    mh = mm.metrics.create()
    # 메트릭 계산
    summary = mh.compute(acc, metrics=METRICS)
    # Tracker 폴더 이름으로 인덱스 설정
    summary.index = [os.path.basename(track_folder)]
    return summary, mh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt',    required=True, help='GT label_02 폴더')
    parser.add_argument('--track', required=True, help='Tracker 결과 폴더')
    parser.add_argument('--iou',   type=float, default=0.5, help='IoU threshold')
    args = parser.parse_args()

    summary, mh = evaluate_tracker(args.gt, args.track, args.iou)
    print(f"\n=== Summary for {os.path.basename(args.track)} ===")
    print(mm.io.render_summary(summary))

    # CSV 저장
    out_csv = f"{os.path.basename(args.track)}_metrics.csv"
    summary.to_csv(out_csv)
    print(f"Saved metrics to {out_csv}")

if __name__ == '__main__':
    main()
