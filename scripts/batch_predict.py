import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# PATHS
# =========================
MODEL_PATH = "models/efficientnet.keras"
TEST_DIR = "dataset/clean_dataset/test"
CLASS_PATH = "deployment/class_names.txt"
OUTPUT_CSV = "outputs/predictions/batch_results.csv"

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD CLASS NAMES
# =========================
with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# STORAGE
# =========================
results = []

# =========================
# LOOP THROUGH ALL CLASSES
# =========================
for true_class in os.listdir(TEST_DIR):
    class_folder = os.path.join(TEST_DIR, true_class)

    if not os.path.isdir(class_folder):
        continue

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)

        try:
            # Load image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediction
            preds = model.predict(img_array, verbose=0)
            pred_index = np.argmax(preds[0])
            pred_class = class_names[pred_index]
            confidence = float(np.max(preds[0]))

            # Save result
            results.append([
                img_path,
                true_class,
                pred_class,
                confidence,
                true_class == pred_class
            ])

            print(f"Processed: {img_path}")

        except Exception as e:
            print(f"Error with {img_path}: {e}")

# =========================
# SAVE CSV
# =========================
os.makedirs("outputs/predictions", exist_ok=True)

df = pd.DataFrame(results, columns=[
    "image",
    "true_class",
    "predicted_class",
    "confidence",
    "correct"
])

df.to_csv(OUTPUT_CSV, index=False)

# =========================
# ACCURACY
# =========================
accuracy = df["correct"].mean()

print("\n======================")
print("BATCH TEST COMPLETE")
print("Total Images:", len(df))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("CSV Saved:", OUTPUT_CSV)
print("======================\n")