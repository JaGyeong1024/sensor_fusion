# to_kitti_format.py
import os

def yolo_to_kitti_bbox(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w/2)*img_w
    y1 = (yc - h/2)*img_h
    x2 = (xc + w/2)*img_w
    y2 = (yc + h/2)*img_h
    return x1, y1, x2, y2

def save_pred_dir_to_kitti(preds_dir, output_dir, img_size):
    os.makedirs(output_dir, exist_ok=True)
    class_map = {0:'Car',1:'Van',2:'Cyclist',3:'DontCare'}
    files = sorted(f for f in os.listdir(preds_dir) if f.endswith('.txt'))
    prev = None
    for fname in files:
        fid = int(os.path.splitext(fname)[0])
        if prev is not None and fid!=prev+1:
            print(f"Warning: missing frames {prev+1:06d}–{fid-1:06d}")
        prev = fid
        out_path = os.path.join(output_dir, f'{fid:06d}.txt')
        with open(os.path.join(preds_dir, fname)) as fr, open(out_path,'w') as fw:
            for line in fr:
                parts = line.strip().split()
                cls = int(parts[0]); xc,yc,w,h,sc = map(float, parts[1:6])
                x1,y1,x2,y2 = yolo_to_kitti_bbox(xc,yc,w,h,*img_size)
                cls_str = class_map.get(cls,'DontCare')
                fw.write(f"{cls_str} 0 0 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
                         f" 0 0 0 0 0 0 0 {sc:.4f}\n")
