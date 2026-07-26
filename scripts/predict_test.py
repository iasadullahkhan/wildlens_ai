import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

from tensorflow.keras.applications.efficientnet import preprocess_input
# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/efficientnet.keras"
CLASS_PATH = "deployment/class_names.txt"
TEST_IMAGE = r"dataset/clean_dataset/test/common_leopard/common_leopard_015.jpg"

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load class names
# -----------------------------
with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -----------------------------
# Load and preprocess image
# -----------------------------
img = image.load_img(TEST_IMAGE, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
confidence = float(np.max(predictions[0]))

predicted_class = class_names[predicted_index]

# -----------------------------
# Output
# -----------------------------
print("\n======================")
print("Image:", TEST_IMAGE)
print("Predicted Class:", predicted_class)
print("Confidence:", round(confidence * 100, 2), "%")
print("======================\n")

# -----------------------------
# Save result
# -----------------------------
os.makedirs("outputs/predictions", exist_ok=True)

output_file = "outputs/predictions/result.txt"
with open(output_file, "w") as f:
    f.write(f"Image: {TEST_IMAGE}\n")
    f.write(f"Predicted Class: {predicted_class}\n")
    f.write(f"Confidence: {confidence:.4f}\n")

print("Result saved to:", output_file)