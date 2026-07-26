"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)

Generates saliency heatmaps that highlight which spatial regions of an input
image most strongly influenced the model's top predicted class.

Reference:
    Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (https://arxiv.org/abs/1610.02391)
"""

import tensorflow as tf
import numpy as np


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_search_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Returns the innermost sub-model to search for Conv2D layers.

    For transfer-learning architectures (e.g. EfficientNet wrapped in a
    functional model), the conv layers live inside a nested base model.
    Walking one level down is sufficient for all standard Keras wrappers.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return model


def _find_last_conv_layer(
    model: tf.keras.Model,
) -> tuple[tf.keras.Model, tf.keras.layers.Layer]:
    """
    Locates the final Conv2D layer inside *model* (or its first sub-model).

    Forcing a dummy forward pass before the search ensures the graph is fully
    built, which is required for layer introspection to work reliably.

    Args:
        model: A compiled or un-compiled Keras model.

    Returns:
        A (search_model, last_conv_layer) tuple.

    Raises:
        ValueError: If no Conv2D layer is found anywhere in the model.
    """
    # Ensure the computation graph is materialised.
    _ = model(tf.zeros((1, 224, 224, 3)))

    search_model = _resolve_search_model(model)

    for layer in reversed(search_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return search_model, layer

    raise ValueError(
        "No Conv2D layer found in the model. "
        "Grad-CAM requires at least one convolutional layer."
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for the top-predicted class.

    The heatmap is normalised to [0, 1] and has the spatial dimensions of
    the last conv layer's feature map (typically 7×7 for EfficientNet at
    224×224 input). Resize to the original image resolution before overlaying.

    Args:
        img_array:  Float32 array of shape (1, H, W, 3), values in [0, 255].
        model:      Loaded Keras model whose architecture contains Conv2D layers.

    Returns:
        A 2-D float32 NumPy array in [0, 1] representing per-pixel importance.
    """
    search_model, last_conv = _find_last_conv_layer(model)

    # Warm the graph so shape inference is stable before building the sub-model.
    model(tf.convert_to_tensor(img_array))

    # Build a twin model that exposes both the target conv output and the
    # softmax head — needed so GradientTape can link them.
    grad_model = tf.keras.Model(
        inputs=search_model.input,
        outputs=[last_conv.output, search_model.output],
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        # Scalar loss = logit of the winning class.
        loss = tf.gather(predictions[0], class_idx)

    # Gradients of the class score w.r.t. the conv feature map.
    grads = tape.gradient(loss, conv_outputs)            # (1, h, w, C)

    # Global-average-pool the gradients over the spatial dimensions → (C,).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each channel of the feature map by its pooled gradient.
    conv_outputs = conv_outputs[0]                       # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                        # (h, w)

    # ReLU: keep only activations that positively influence the class score.
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalise to [0, 1]; guard against all-zero maps.
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()