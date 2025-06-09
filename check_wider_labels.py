# check_wider_labels.py
import random, cv2, scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt

mat_path = Path("WIDER_train/wider_face_split/wider_face_train.mat")
img_root = Path("WIDER_train/images")

mat     = sio.loadmat(mat_path)
events  = mat["event_list"]
files   = mat["file_list"]
boxes   = mat["face_bbx_list"]

# 1) Zufälliges Event/Bild
ev_idx   = random.randrange(len(events))
evt_name = events[ev_idx][0][0]                 # ← fix

file_list = files [ev_idx][0]
box_list  = boxes[ev_idx][0]

fi_idx   = random.randrange(len(file_list))
img_stem = file_list[fi_idx][0][0]              # ← fix
bbxs     = box_list [fi_idx][0]

# 2) Passende Extension suchen (.jpg/.jpeg/.png)
for ext in (".jpg", ".jpeg", ".png"):
    img_p = img_root / evt_name / f"{img_stem}{ext}"
    if img_p.exists():
        break
else:
    raise FileNotFoundError(
        f"Bild nicht gefunden:\n{img_root / evt_name / img_stem}.[jpg|jpeg|png]"
    )

# 3) Bild laden & Boxen zeichnen
img = cv2.imread(str(img_p))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for (x, y, w, h) in bbxs:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(6, 8))
plt.title(img_p.name)
plt.imshow(img); plt.axis("off"); plt.tight_layout(); plt.show()
