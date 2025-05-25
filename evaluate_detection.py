# evaluate_detection.py

"""
TrackEval 기반 검출 성능 평가 스크립트
사용법:
python3 evaluate_detection.py \
  --pred_dirs eval_detection/vishal eval_detection/udacity eval_detection/sensorfusion \
  --gt_dir   training/label_02 \
  --out_csv  detection_results.csv
"""

import os
import csv
import argparse
import trackeval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dirs', nargs='+', required=True,
                        help='KITTI 형식 예측 결과 폴더 목록')
    parser.add_argument('--gt_dir',    required=True,
                        help='KITTI GT 라벨 폴더 (training/label_02)')
    parser.add_argument('--out_csv',   required=True,
                        help='출력 CSV 파일 경로')
    args = parser.parse_args()

    # TrackEval 설정: PrecisionRecall (mAP) 모듈
    evaluator = trackeval.Evaluator(metrics=['PrecisionRecall'],
                                    output_folder='eval_detection')

    # 공통 Config
    cfg = {
        'GT_FOLDER': args.gt_dir,
        'TRACKERS_FOLDER': None,  # 아래 for문에서 채울 예정
        'SEQMAP_FILE':  'TrackEval/scripts/seqmaps/kitti_object.txt',
        'GT_LOC_FORMAT':       '{}/{:s}.txt',
        'TRACKER_LOC_FORMAT':  '{}/{:s}.txt',
        'USE_PARALLEL':        False,
        'PRINT_CONFIG':        False,
        'PRINT_RESULTS':       False
    }

    # 결과 CSV 헤더 작성
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module','mAP@0.5','mAP@0.75'])

        # 각 모듈별 평가
        for pred_dir in args.pred_dirs:
            cfg['TRACKERS_FOLDER'] = pred_dir
            # KITTI_OBJECT 데이터셋 로드
            dataset = trackeval.datasets.KITTI_OBJECT(cfg)
            # 평가 실행
            metrics, _ = evaluator.evaluate([dataset])

            # 결과에서 PrecisionRecall 지표 가져오기
            pr = metrics['KITTI_OBJECT']['PrecisionRecall']
            ap50 = pr['ap_50']   # mAP@0.5
            ap75 = pr['ap_75']   # mAP@0.75

            writer.writerow([os.path.basename(pred_dir), f"{ap50:.3f}", f"{ap75:.3f}"])
            print(f"[결과] {os.path.basename(pred_dir)}: mAP@0.5={ap50:.3f}, mAP@0.75={ap75:.3f}")

    print(f"\n완료: 검출 성능이 '{args.out_csv}'에 저장되었습니다.")

if __name__ == '__main__':
    main()
