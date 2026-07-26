import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

# =====================================================
# 1. LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(
    r"F:\Final Year Project\01. Final Year Project\models\efficientnet.keras",
    compile=False
)

_ = model(tf.zeros((1, 224, 224, 3)))

# =====================================================
# 2. DATASET PATH
# =====================================================
test_dir = r"F:\Final Year Project\01. Final Year Project\dataset\clean_dataset\test"

output_dir = r"F:\Final Year Project\01. Final Year Project\outputs\gradcam"
os.makedirs(output_dir, exist_ok=True)

# =====================================================
# 3. GET CLASSES
# =====================================================
classes = sorted(os.listdir(test_dir))

# =====================================================
# 4. FIND LAST CONV LAYER
# =====================================================
last_conv_layer = None
for layer in reversed(model.layers):
    if len(layer.output.shape) == 4:
        last_conv_layer = layer
        break

print("Last Conv Layer:", last_conv_layer.name)

# =====================================================
# 5. LOOP OVER ALL CLASSES
# =====================================================
for class_name in classes:

    class_path = os.path.join(test_dir, class_name)
    images = os.listdir(class_path)

    # pick random image per class
    img_name = random.choice(images)
    img_path = os.path.join(class_path, img_name)

    print("Processing:", class_name, img_name)

    # =================================================
    # PREPROCESS
    # =================================================
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # =================================================
    # GRADIENT TAPE
    # =================================================
    with tf.GradientTape() as tape:
        conv_output = last_conv_layer(img_array)
        tape.watch(conv_output)

        x = conv_output
        for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
            x = layer(x)

        predictions = x
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # =================================================
    # PROCESS HEATMAP
    # =================================================
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    # =================================================
    # OVERLAY
    # =================================================
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    # =================================================
    # SAVE
    # =================================================
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    save_path = os.path.join(class_output_dir, f"gradcam_{img_name}")
    cv2.imwrite(save_path, overlay)

print("DONE: Grad-CAM generated for all classes")