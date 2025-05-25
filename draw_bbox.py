import cv2, glob
import os

preds = 'preds_vishal'
imgs  = 'training/image_02'
out_viz = 'viz'
os.makedirs(out_viz, exist_ok=True)

for i, img_path in enumerate(sorted(glob.glob(f"{imgs}/*.png"))[:5]):  # 처음 5장만
    fn = os.path.basename(img_path)[:-4]
    lbl = os.path.join(preds, f"{fn}.txt")
    im  = cv2.imread(img_path)
    if not os.path.exists(lbl): continue
    with open(lbl) as f:
        for line in f:
            cls, x1,y1,x2,y2,score = line.split()
            x1,y1,x2,y2 = map(int,[float(x1),float(y1),float(x2),float(y2)])
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(im, f"{cls}:{float(score):.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
    cv2.imwrite(os.path.join(out_viz, fn+'.png'), im)
