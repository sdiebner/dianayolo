# detect.py – Inferenz mit optionaler Debug-Ausgabe
import argparse, os, cv2, torch, numpy as np
from pathlib import Path
from models.yolo import YOLOv5Small
from utils.boxes import non_max_suppression

# ------------------------------------------------------------------------
def letterbox(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r   = min(new_shape[0] / h, new_shape[1] / w)
    new = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape[1] - new[0]) / 2, (new_shape[0] - new[1]) / 2
    img = cv2.resize(img, new, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, int(dh - 0.1), int(dh + 0.1),
                                   int(dw - 0.1), int(dw + 0.1),
                                   cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# ------------------------------------------------------------------------
def load_model(ckpt, dev):
    m = YOLOv5Small(num_classes=1)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    print("Anchors:", m.anchors)                       # Debug-Ausgabe
    return m.to(dev).eval()

# ------------------------------------------------------------------------
@torch.no_grad()
def predict(model, img_bgr, dev, size=640, conf=0.25, iou=0.45, debug=False):
    # -------- Preprocess --------------------------------------------------
    img, r, (dw, dh) = letterbox(img_bgr, size)
    img = img[:, :, ::-1].copy()                       # BGR→RGB + copy()
    img = torch.from_numpy(img.transpose(2, 0, 1))     # HWC→CHW
    img = img.float().div(255).unsqueeze(0).to(dev)    # (1,3,H,W)

    # -------- Forward -----------------------------------------------------
    strides = [4, 8, 16]
    preds   = []
    for li, p in enumerate(model(img)):
        bs, _, ny, nx = p.shape
        p = p.view(bs, 3, model.num_outputs, ny, nx).permute(0,1,3,4,2).contiguous()

        if debug:
            obj = p[..., 4].sigmoid()
            print(f"P{li+3} obj-min/max: {obj.min():.3f} / {obj.max():.3f}")

        gy, gx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        grid   = torch.stack((gx, gy), 2).view(1,1,ny,nx,2).to(dev)
        anchor = model.anchors[li].to(dev).view(1,3,1,1,2)

        xy  = (p[..., :2].sigmoid()*2 - 0.5 + grid) * strides[li]
        wh  = ((p[..., 2:4].sigmoid()*2)**2) * anchor * strides[li]
        obj =  p[..., 4:5].sigmoid()
        cls =  p[..., 5:].sigmoid()

        preds.append(torch.cat((xy, wh, obj, cls), -1).view(bs, -1, model.num_outputs))

    det = non_max_suppression(torch.cat(preds, 1), conf, iou)[0].cpu().numpy()

    # -------- Undo Letterbox ---------------------------------------------
    if debug: print("before undo:", det[:3, :5])
    if det.shape[0]:
        det[:, :4] -= [dw, dh, dw, dh]
        det[:, :4] /= r
        det[:, :4]  = det[:, :4].round()
    if debug: print("after  undo:", det[:3, :5])
    return det

# ------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--weights", default="weights/best_model.pth")
    pa.add_argument("--source",  default="demo.jpg")
    pa.add_argument("--img_size", type=int, default=640)
    pa.add_argument("--conf",     type=float, default=0.25)
    pa.add_argument("--iou",      type=float, default=0.45)
    pa.add_argument("--debug",    action="store_true")
    args = pa.parse_args()

    dev   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.weights, dev)

    src  = Path(args.source)
    imgs = [src] if src.is_file() else list(src.glob("*.*"))
    os.makedirs("runs/detect", exist_ok=True)

    for p in imgs:
        im  = cv2.imread(str(p))
        det = predict(model, im, dev, args.img_size, args.conf, args.iou, args.debug)
        for x1,y1,x2,y2,cf,_ in det:
            cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(im,f"{cf:.2f}",(int(x1),int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        out = f"runs/detect/{p.stem}_det.jpg"
        cv2.imwrite(out, im); print("saved →", out)

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
