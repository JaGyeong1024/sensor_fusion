# run_mid_fusion.py
import os
import glob
import argparse

def iou(a, b):
    # a, b = (x1,y1,x2,y2)
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def run_mid_fusion(img_dir, ldr_dir, out_dir, iou_thr=0.3):
    os.makedirs(out_dir, exist_ok=True)
    for ip in sorted(glob.glob(os.path.join(img_dir, '*.txt'))):
        fid = os.path.basename(ip)
        imgs = [list(map(float,l.split())) for l in open(ip)]
        ldr_file = os.path.join(ldr_dir, fid)
        ldrs = os.path.exists(ldr_file) and [list(map(float,l.split())) for l in open(ldr_file)] or []
        fused = []
        for cls1,x1,y1,x2,y2,s1 in imgs:
            for cls2,xx1,yy1,xx2,yy2,s2 in ldrs:
                if iou((x1,y1,x2,y2),(xx1,yy1,xx2,yy2))>iou_thr:
                    sc = 0.5*s1 + 0.5*(s2/len(ldrs)) if ldrs else s1
                    fused.append((cls1,x1,y1,x2,y2,sc))
        with open(os.path.join(out_dir,fid),'w') as f:
            for c,x1,y1,x2,y2,sc in fused:
                f.write(f"{c} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {sc:.4f}\n")
    print("완료: Mid-level 융합 →", out_dir)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--img_preds',   required=True)
    p.add_argument('--lidar_preds', required=True)
    p.add_argument('--out_dir',     required=True)
    args = p.parse_args()
    run_mid_fusion(args.img_preds, args.lidar_preds, args.out_dir)