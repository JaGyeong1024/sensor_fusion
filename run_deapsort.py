# run_deepsort.py
import os
from deep_sort import DeepSort

def run_deepsort(det_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    deepsort = DeepSort()
    for fname in sorted(os.listdir(det_dir)):
        fid = int(fname.split('.')[0])
        dets=[]
        with open(os.path.join(det_dir,fname)) as f:
            for l in f:
                p=l.strip().split()
                x1,y1,x2,y2,sc = map(float,p[4:9])
                dets.append([x1,y1,x2,y2,sc])
        outputs = deepsort.update(dets)
        with open(f"{out_dir}/{fid:06d}.txt",'w') as fo:
            for bbox, tid in outputs:
                x1,y1,x2,y2 = bbox
                fo.write(f"{fid} {tid} -1 -1 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} 0 0 0 0\n")
    print("DeepSORT tracking →", out_dir)
