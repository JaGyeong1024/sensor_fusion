# run_image_detect.py
# YOLOv5 custom 모델로 순수 이미지 검출만 수행
import os
import glob
import argparse
import torch

def run_image_detect(kroot, model_path, preds_dir, img_size):
    os.makedirs(preds_dir, exist_ok=True)
    # YOLOv5 custom weight 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    img_files = sorted(glob.glob(os.path.join(kroot, 'image_02', '*.png')))
    for img_path in img_files:
        fid = os.path.splitext(os.path.basename(img_path))[0]
        results = model(img_path, size=img_size)
        out_file = os.path.join(preds_dir, f"{fid}.txt")
        with open(out_file, 'w') as f:
            for *xyxy, conf, cls in results.xyxy[0].tolist():
                x1,y1,x2,y2 = xyxy
                f.write(f"{int(cls)} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.4f}\n")
    print("완료: 이미지 검출 →", preds_dir)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kitti_root', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--preds_dir',  required=True)
    p.add_argument('--img_size',   nargs=2, type=int, default=[1242,375])
    args = p.parse_args()
    run_image_detect(
        args.kitti_root,
        args.model_path,
        args.preds_dir,
        tuple(args.img_size)
    )