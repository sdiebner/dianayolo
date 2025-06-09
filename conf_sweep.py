#!/usr/bin/env python
# conf_sweep.py – finde optimale Confidence-Schwelle & zeichne Kurve
# ---------------------------------------------------------------
import argparse, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from datasets.wider_face import WIDERFaceDataset
from torch.utils.data    import DataLoader
from models.yolo         import YOLOv5Small
from utils.boxes         import (non_max_suppression,
                                 xywh2xyxy, xyxy2xywh, bbox_iou)
from eval_metrics        import ap_per_class            # schon vorhanden

# -----------------------------------------------------------------
def build_val_loader(img_size, bs=8):
    val_ds = WIDERFaceDataset("WIDER_val/images",
                              "WIDER_train/wider_face_split/wider_face_val.mat",
                              img_size=img_size)
    collate = lambda batch: (list(zip(*batch))[0], list(zip(*batch))[1])
    return DataLoader(val_ds, batch_size=bs, shuffle=False,
                      num_workers=0, collate_fn=collate)

@torch.no_grad()
def precision_recall(model, dataloader, conf_th, strides, img_size, device):
    stats = []
    for imgs, tgt_list in tqdm(dataloader, leave=False):
        imgs = torch.stack(imgs).to(device)

        # --- Predictions dekodieren (wie im Training/Eval) ---
        preds = []
        for li, p in enumerate(model(imgs)):
            bs,_,ny,nx = p.shape
            p = p.view(bs,3,model.num_outputs,ny,nx).permute(0,1,3,4,2).contiguous()
            gy,gx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
            grid   = torch.stack((gx,gy),2).view(1,1,ny,nx,2).to(device)
            anchor = model.anchors[li].to(device).view(1,3,1,1,2)
            xy  = (p[...,:2].sigmoid()*2 - 0.5 + grid) * strides[li]
            wh  = ((p[...,2:4].sigmoid()*2)**2) * anchor * strides[li]
            obj =  p[...,4:5].sigmoid()
            cls =  p[...,5:].sigmoid()
            preds.append(torch.cat((xy,wh,obj,cls),-1).view(bs,-1,model.num_outputs))
        dets = non_max_suppression(torch.cat(preds,1), conf_th, 0.45)

        # --- Targets zusammenstellen ---
        tgt = [torch.cat((torch.full((t.shape[0],1),i), t),1)
               for i,t in enumerate(tgt_list) if t.numel()]
        targets = torch.cat(tgt,0).to(device) if tgt else torch.zeros((0,6), device=device)

        # --- pro Bild stats sammeln ---
        for si, det in enumerate(dets):
            gt = targets[targets[:,0]==si, 1:]
            correct = torch.zeros(det.shape[0], dtype=torch.bool, device=device)
            if gt.numel():
                gt_boxes = xywh2xyxy(gt[:,1:5] * img_size)
                for di,(*box_xyxy,conf,cls_id) in enumerate(det):
                    iou = bbox_iou(
                        xyxy2xywh(torch.tensor(box_xyxy, device=device)).unsqueeze(0),
                        xyxy2xywh(gt_boxes)
                    ).max(1)[0]
                    correct[di] = (iou > 0.5)
            stats.append((correct.cpu(), det[:,4].cpu(),
                          det[:,5].cpu().int(),
                          gt[:,0].cpu().int() if gt.numel() else torch.tensor([],dtype=torch.int)))
    if not stats:
        return 0.,0.,0.
    tp, conf, pred_cls, target_cls = [torch.cat(x,0).numpy() for x in zip(*stats)]
    P,R,_ = ap_per_class(tp, conf, pred_cls, target_cls)
    F1 = 2*P*R / (P+R+1e-9)
    return P,R,F1

# -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--min_conf", type=float, default=0.001)
    ap.add_argument("--max_conf", type=float, default=0.5)
    ap.add_argument("--steps",   type=int,   default=40)
    args = ap.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    model  = YOLOv5Small(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    strides = [4,8,16]
    val_dl  = build_val_loader(args.img_size, bs=8)

    conf_grid = np.linspace(args.min_conf, args.max_conf, args.steps)
    P,R,F1 = [],[],[]
    print(f"⇢ Sweep {len(conf_grid)} Schwellen …")
    for c in conf_grid:
        p,r,f = precision_recall(model, val_dl, c, strides, args.img_size, device)
        P.append(p); R.append(r); F1.append(f)
        print(f"conf={c:7.4f} | P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    best_i = int(np.argmax(F1))
    print("\n✔  Beste Schwelle nach F1:",
          f"{conf_grid[best_i]:.4f}  (P={P[best_i]:.3f}, R={R[best_i]:.3f})")

    # ---------- Plot ----------
    plt.plot(conf_grid, P,  label="Precision")
    plt.plot(conf_grid, R,  label="Recall")
    plt.plot(conf_grid, F1, label="F1")
    plt.axvline(conf_grid[best_i], ls="--", color="grey",
                label=f"best={conf_grid[best_i]:.3f}")
    plt.xlabel("Confidence-Threshold"); plt.ylabel("Score")
    plt.legend(); plt.tight_layout()
    plt.savefig("conf_sweep.png", dpi=150)
    print("Plot gespeichert  →  conf_sweep.png")

if __name__ == "__main__":
    main()
