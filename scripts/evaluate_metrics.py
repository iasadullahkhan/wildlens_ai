import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/efficientnet.keras")

# =========================
# CLASS NAMES
# =========================
with open("deployment/class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# DATASET PATH
# =========================
test_dir = "dataset/clean_dataset/test"

y_true = []
y_pred = []

# =========================
# LOOP DATA
# =========================
for true_class in os.listdir(test_dir):
    class_path = os.path.join(test_dir, true_class)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array, verbose=0)
        pred_class = class_names[np.argmax(preds)]

        y_true.append(true_class)
        y_pred.append(pred_class)

# =========================
# REPORT
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

df = pd.DataFrame(report).transpose()
df.to_csv("outputs/classification_report.csv")

print("Classification report saved!")




import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred, labels=class_names)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - EfficientNet")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig("outputs/confusion_matrix.png")
plt.show()