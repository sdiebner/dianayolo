# train.py – YOLOv5Small Face-Training mit vollständigem Checkpointing
# -------------------------------------------------------------------
import os, csv, math, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.wider_face import WIDERFaceDataset
from models.yolo        import YOLOv5Small
from eval_metrics       import evaluate              # AP50-Eval

# ----------------------- Helper ------------------------------------------------
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

def get_device():
    return torch.device(
        "mps"  if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu")

def build_target_tensor(targets, device):
    out = []
    for i, t in enumerate(targets):
        if t.numel():
            t = t.to(device)
            img_idx = torch.full((t.shape[0], 1), i, device=device, dtype=t.dtype)
            out.append(torch.cat((img_idx, t), 1))
    return torch.cat(out, 0) if out else torch.zeros((0, 6), device=device)

# ----------------------- Target-Zuordnung --------------------------------------
def build_targets(preds, targets, anchors, strides):
    tcls, tbox, indices, anch = [], [], [], []
    nl, na = len(preds), anchors.shape[1]
    gain   = torch.ones(6, device=targets.device)

    if not targets.shape[0]:
        for _ in range(nl):
            indices.append((torch.tensor([]),)*4)
            tbox.append(torch.tensor([])); anch.append(torch.tensor([])); tcls.append(torch.tensor([]))
        return tcls, tbox, indices, anch

    for i in range(nl):
        gain[2:] = torch.tensor(preds[i].shape, device=targets.device)[[3,2,3,2]]
        t = targets * gain
        nt = t.shape[0]
        t  = t.repeat(na, 1)
        ai = torch.arange(na, device=targets.device).repeat_interleave(nt)

        a_wh = anchors[i][ai]
        r = t[:, 4:6] / a_wh
        j = (torch.max(r, 1. / r).max(1)[0] < 4.0)
        if not j.any():
            indices.append((torch.tensor([]),)*4)
            tbox.append(torch.tensor([])); anch.append(torch.tensor([])); tcls.append(torch.tensor([]))
            continue

        t, a_wh, ai = t[j], a_wh[j], ai[j]
        b, cls = t[:, :2].long().T
        gxy, gwh = t[:, 2:4], t[:, 4:6]
        gij = gxy.long(); gi, gj = gij.T

        indices.append((b, ai, gj, gi))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(a_wh)
        tcls.append(cls)
    return tcls, tbox, indices, anch

# ----------------------- IoU & Loss --------------------------------------------
def bbox_iou(box1, box2, eps=1e-7):
    (x1,y1,w1,h1),(x2,y2,w2,h2) = box1.T, box2.T
    inter = (torch.min(x1+w1/2, x2+w2/2) - torch.max(x1-w1/2, x2-w2/2)).clamp_(0) * \
            (torch.min(y1+h1/2, y2+h2/2) - torch.max(y1-h1/2, y2-h2/2)).clamp_(0)
    union = w1*h1 + w2*h2 - inter + eps
    return inter / union

def compute_loss(preds, targets, model, strides):
    device = preds[0].device
    BCE, MSE = nn.BCEWithLogitsLoss(), nn.MSELoss()
    anchors = model.anchors.to(device)
    tcls, tbox, indices, anch = build_targets(preds, targets, anchors, strides)
    lbox = lobj = lcls = torch.zeros(1, device=device)

    for i, p in enumerate(preds):
        bs, _, ny, nx = p.shape
        p = p.view(bs,3,model.num_outputs,ny,nx).permute(0,1,3,4,2).contiguous()
        b,a,gj,gi = indices[i]
        tobj = torch.zeros_like(p[...,0])

        if b.numel():
            pxy = (p[b,a,gj,gi,:2].sigmoid()*2 - 0.5)
            pwh = ((p[b,a,gj,gi,2:4].sigmoid()*2)**2) * anch[i]
            pbox = torch.cat((pxy,pwh),1)

            lbox += 0.05 * MSE(pbox, tbox[i])
            iou  = bbox_iou(pbox, tbox[i]).detach()
            tobj[b,a,gj,gi] = iou.clamp_(0)

            if model.num_classes > 1:
                tc = torch.zeros_like(p[b,a,gj,gi,5:])
                tc[range(b.shape[0]), tcls[i]] = 1.
                lcls += 0.5 * BCE(p[b,a,gj,gi,5:], tc)
        lobj += 1.0 * BCE(p[...,4], tobj)
    return lbox + lobj + lcls

# ----------------------- Haupt-Training ----------------------------------------
def train(args):
    device = get_device(); print("Using device:", device)
    model  = YOLOv5Small(num_classes=1).to(device)

    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    train_ds = WIDERFaceDataset("WIDER_train/images",
                                "WIDER_train/wider_face_split/wider_face_train.mat",
                                img_size=args.img_size)
    val_ds   = WIDERFaceDataset("WIDER_val/images",
                                "WIDER_train/wider_face_split/wider_face_val.mat",
                                img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=collate_fn)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    warm = 5
    lf = lambda x: (x / warm) if x < warm else 0.5*(1+math.cos(math.pi*(x-warm)/(args.epochs-warm)))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lf)

    strides = torch.tensor([4,8,16], device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "train_log.csv")
    with open(csv_path,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","mAP50"])

    # ----- Tracking best scores
    best_val, best_map = 1e9, 0.0
    patience, bad = 35, 0
    save_all_ckpt = True          # <-- hier auf False stellen, wenn du NICHT jede Epoche speichern willst
    map_interval  = 5             # alle n Epochen mAP berechnen

    tl_hist, vl_hist, map_hist = [], [], []

    for epoch in range(args.epochs):

        # ---------- TRAIN ----------
        model.train(); tloss = 0.
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=80)
        for imgs, targets in pbar:
            imgs = torch.stack(imgs).to(device)
            tgt  = build_target_tensor(targets, device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    preds = model(imgs)
                    loss  = compute_loss(preds, tgt, model, strides)
            else:
                preds = model(imgs)
                loss  = compute_loss(preds, tgt, model, strides)

            opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

            tloss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}")

        tl_hist.append(tloss/len(train_dl))

        # ---------- VAL ----------
        model.eval(); vloss = 0.
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs = torch.stack(imgs).to(device)
                tgt  = build_target_tensor(targets, device)
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        preds = model(imgs); loss = compute_loss(preds, tgt, model, strides)
                else:
                    preds = model(imgs); loss = compute_loss(preds, tgt, model, strides)
                vloss += loss.item()
        vloss /= len(val_dl); vl_hist.append(vloss)

        # ---------- mAP ----------
        mAP50 = "-"
        if (epoch+1) % map_interval == 0:
            met = evaluate(model, val_dl, device, img_size=args.img_size)
            mAP50 = f"{met['mAP50']:.3f}"
            map_hist.append(met['mAP50'])
            print(f"mAP50 {mAP50} | P {met['P']:.3f} | R {met['R']:.3f}")

            # best-mAP-Checkpoint
            if met['mAP50'] > best_map + 1e-4:
                best_map = met['mAP50']
                torch.save(model.state_dict(), os.path.join(args.output_dir,"best_map.pth"))
                print("✓   best mAP-model updated")

        # ---------- Logging ----------
        print(f"Epoch {epoch+1}/{args.epochs} | train {tl_hist[-1]:.3f} | "
              f"val {vloss:.3f} | mAP50 {mAP50}")
        with open(csv_path,"a",newline="") as f:
            csv.writer(f).writerow([epoch+1, tl_hist[-1], vloss, mAP50])

        # ---------- Checkpoints ----------
        if save_all_ckpt:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.pth"))

        if vloss < best_val - 1e-3:
            best_val, bad = vloss, 0
            torch.save(model.state_dict(), os.path.join(args.output_dir,"best_val.pth"))
            print("✓   best val-loss updated")
        else:
            bad += 1

        if bad >= patience:
            print(f"⏹ Early-stopped (patience {patience}) at epoch {epoch+1}")
            break

        scheduler.step()

    # ---------- Loss-Curve ----------
    plt.plot(tl_hist,label="train"); plt.plot(vl_hist,label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,"loss_curve.png"))
    print("Loss curve saved →", os.path.join(args.output_dir,"loss_curve.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=200)
    ap.add_argument("--batch_size", type=int,   default=8)
    ap.add_argument("--img_size",   type=int,   default=640)
    ap.add_argument("--lr",         type=float, default=0.01)
    ap.add_argument("--output_dir",           default="weights")
    train(ap.parse_args())
