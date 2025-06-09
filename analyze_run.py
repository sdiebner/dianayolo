import pandas as pd, matplotlib.pyplot as plt, torch, cv2
from pathlib import Path
from detect import load_model, predict   # nutzt deine vorhandene detect.py

# ---------- Losskurve anzeigen ----------
log = pd.read_csv("weights/train_log.csv")
plt.figure(figsize=(6,4))
plt.plot(log.epoch, log.train_loss, label="train")
plt.plot(log.epoch, log.val_loss, label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
plt.show()

# ---------- Beispiel-Detektionen ----------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model  = load_model("weights/best_model.pth", device)
sample_dir = Path("WIDER_val/images")
out_dir    = Path("runs/presentation"); out_dir.mkdir(parents=True, exist_ok=True)

for img_path in list(sample_dir.rglob("*.jpg"))[:6]:
    img = cv2.imread(str(img_path))
    det = predict(model, img, device, 512, 0.25, 0.45)
    for x1,y1,x2,y2,conf,_ in det:
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    cv2.imwrite(str(out_dir / img_path.name), img)

print("✓ 6 Beispielbilder mit Boxen gespeichert unter", out_dir)
