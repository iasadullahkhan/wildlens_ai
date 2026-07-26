import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# =====================================================
# 1. LOAD MODEL
# =====================================================
model = tf.keras.models.load_model(
    r"F:\Final Year Project\01. Final Year Project\models\efficientnet.keras",
    compile=False
)

# Force build
_ = model(tf.zeros((1, 224, 224, 3)))

# =====================================================
# 2. IMAGE
# =====================================================
img_path = r"F:\Final Year Project\01. Final Year Project\dataset\clean_dataset\test\common_leopard\common_leopard_015.jpg"

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# =====================================================
# 3. FIND LAST CONV LAYER OUTPUT DIRECTLY
# =====================================================
last_conv_layer = None
for layer in reversed(model.layers):
    if len(layer.output.shape) == 4:
        last_conv_layer = layer
        break

print("Last Conv Layer:", last_conv_layer.name)

# =====================================================
# 4. GRADIENT TAPE (DIRECT METHOD - NO FUNCTIONAL API)
# =====================================================
with tf.GradientTape() as tape:
    conv_output = last_conv_layer(img_array)
    tape.watch(conv_output)

    # forward pass after conv layer
    x = conv_output
    for layer in model.layers[model.layers.index(last_conv_layer)+1:]:
        x = layer(x)

    predictions = x
    predicted_class = tf.argmax(predictions[0])
    loss = predictions[:, predicted_class]

# =====================================================
# 5. GRADIENTS
# =====================================================
grads = tape.gradient(loss, conv_output)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_output = conv_output[0]

heatmap = conv_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# =====================================================
# 6. PROCESS HEATMAP
# =====================================================
heatmap = np.maximum(heatmap, 0)

if np.max(heatmap) != 0:
    heatmap = heatmap / np.max(heatmap)

# =====================================================
# 7. OVERLAY
# =====================================================
img_cv = cv2.imread(img_path)
img_cv = cv2.resize(img_cv, (224, 224))

heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

# =====================================================
# 8. SAVE
# =====================================================
output_dir = r"F:\Final Year Project\01. Final Year Project\outputs\gradcam"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "gradcam_common_leopard_015.jpg")
cv2.imwrite(save_path, overlay)

print("Saved Grad-CAM at:", save_path)

# =====================================================
# 9. SHOW
# =====================================================
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM Result")
plt.show()