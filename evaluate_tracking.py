import os
import sys
import argparse

# 로컬 TrackEval 경로를 PYTHONPATH에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
trackeval_path = os.path.join(BASE_DIR, 'TrackEval')
if trackeval_path not in sys.path:
    sys.path.insert(0, trackeval_path)

import trackeval
from trackeval.eval import Evaluator


def evaluate_with_trackeval(track_dir, gt_dir):
    params = {
        'TRACKERS_TO_EVAL': [track_dir],
        'GT_FOLDER': gt_dir,
        'SEQ_INFO_FOLDER': os.path.join('TrackEval', 'data', 'kitti_2d_box', 'seqinfo'),
        'TRACKER_SUB_FOLDER': '',
        'TRACKING_RESULT_EXT': '.txt',
        'METRICS': ['MOTA', 'MOTP', 'IDF1'],
        'PRINT_CONFIG': False
    }
    evaluator = Evaluator.create(params)
    evaluator.evaluate()
    results = evaluator.get_results()
    res = results[track_dir]
    return res['MOTA'], res['MOTP'], res['IDF1']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track_dirs', nargs='+', required=True)
    p.add_argument('--gt_dir',    required=True)
    p.add_argument('--out_csv',   required=True)
    args = p.parse_args()

    with open(args.out_csv, 'w') as fout:
        fout.write('tracker,MOTA,MOTP,IDF1\n')
        for td in args.track_dirs:
            mota, motp, idf1 = evaluate_with_trackeval(td, args.gt_dir)
            name = os.path.basename(td)
            fout.write(f'{name},{mota:.2f},{motp:.2f},{idf1:.2f}\n')
            print(f'[Evaluated] {name}: MOTA={mota:.2f}, MOTP={motp:.2f}, IDF1={idf1:.2f}')

    print(f'완료: 결과가 {args.out_csv}에 저장되었습니다.')

if __name__ == '__main__':
    main()

