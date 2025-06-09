#!/usr/bin/env python
# webcam_detect.py – Live-Face-Detection mit MPS / CUDA oder CPU
# -------------------------------------------------------------
import cv2, torch, numpy as np, argparse, time
from models.yolo  import YOLOv5Small
from utils.boxes  import non_max_suppression

# ----------------------------------- Hilfsfunktionen
def letterbox(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw, dh = dw/2, dh/2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, int(dh-0.1), int(dh+0.1),
                                   int(dw-0.1), int(dw+0.1),
                                   cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

@torch.no_grad()
def infer_one(model, frame, device, conf=0.25, iou=0.45, size=640):
    img0 = frame.copy()
    img, r, (dw, dh) = letterbox(img0, size)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()            # BGR→RGB, HWC→CHW, contiguous!
    img = torch.from_numpy(img).float().div(255).unsqueeze(0).to(device)

    strides = [4, 8, 16]
    preds = []
    for li, p in enumerate(model(img)):
        bs, _, ny, nx = p.shape
        p = p.view(bs, 3, model.num_outputs, ny, nx).permute(0,1,3,4,2).contiguous()
        gy, gx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        grid   = torch.stack((gx, gy), 2).view(1,1,ny,nx,2).to(device)
        anchor = model.anchors[li].to(device).view(1,3,1,1,2)

        xy  = (p[...,:2].sigmoid()*2 - 0.5 + grid) * strides[li]
        wh  = ((p[...,2:4].sigmoid()*2)**2) * anchor * strides[li]
        obj =  p[...,4:5].sigmoid()
        cls =  p[...,5:].sigmoid()
        preds.append(torch.cat((xy, wh, obj, cls), -1).view(bs, -1, model.num_outputs))

    det = non_max_suppression(torch.cat(preds, 1), conf, iou)[0].cpu().numpy()

    if det.shape[0]:
        det[:, 0:4] -= [dw, dh, dw, dh]
        det[:, 0:4] /= r
        det[:, 0:4]  = det[:, 0:4].round()
    return det

# ----------------------------------- Hauptprogramm
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/best_model.pth')
    ap.add_argument('--img_size', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou',  type=float, default=0.45)
    args = ap.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    model  = YOLOv5Small(num_classes=1)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(0)                          # Mac-Cam
    if not cap.isOpened():
        print("❌ Kamera nicht gefunden."); return
    print("⏺  Drücke Q zum Beenden.")

    while True:
        ok, frame = cap.read()
        if not ok: break

        det = infer_one(model, frame, device,
                        conf=args.conf, iou=args.iou, size=args.img_size)

        for x1,y1,x2,y2,cf,_ in det:
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame,f"{cf:.2f}",(int(x1),int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imshow("YOLOv5 Face-Detect (Q=quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
