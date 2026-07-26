import os
import cv2
import matplotlib.pyplot as plt

gradcam_dir = r"F:\Final Year Project\01. Final Year Project\outputs\gradcam"

# ONLY KEEP DIRECTORIES (IMPORTANT FIX)
classes = [d for d in os.listdir(gradcam_dir)
           if os.path.isdir(os.path.join(gradcam_dir, d))]

classes = sorted(classes)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

idx = 0

for class_name in classes[:12]:

    class_path = os.path.join(gradcam_dir, class_name)

    images = os.listdir(class_path)

    if len(images) == 0:
        continue

    img_path = os.path.join(class_path, images[0])
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[idx].imshow(img)
    axes[idx].set_title(class_name, fontsize=9)
    axes[idx].axis("off")

    idx += 1

# turn off unused plots
for j in range(idx, 12):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("outputs/gradcam_grid_clean.png", dpi=300, bbox_inches="tight")
plt.show()