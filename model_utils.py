"""
model_utils.py — Model loading and inference utilities

Handles singleton model loading (with warm-up) and type-safe inference.
The model is loaded once per process and cached globally to avoid
re-initialising TensorFlow's graph on every Streamlit rerun.
"""

import os
import logging

import numpy as np
import tensorflow as tf

# ─── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

_BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR: str = os.path.join(_BASE_DIR, "models")
_MODEL_PATH: str = os.path.join(_MODEL_DIR, "efficientnet.keras")

# ─── Singleton cache ──────────────────────────────────────────────────────────

_model: tf.keras.Model | None = None

# ─── Public API ───────────────────────────────────────────────────────────────

def load_model() -> tf.keras.Model:
    """
    Load the EfficientNet classifier from disk, with graph warm-up.

    The model is cached in module-level state after the first call, so
    subsequent calls are effectively free (O(1) dict lookup).

    Returns:
        The loaded (and warmed-up) ``tf.keras.Model`` instance.

    Raises:
        FileNotFoundError: If the ``.keras`` file does not exist at the
            expected path.
    """
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found.\n"
            f"Expected location: {_MODEL_PATH}\n"
            f"Place 'efficientnet.keras' inside the 'models/' directory."
        )

    logger.info("Loading model from: %s", _MODEL_PATH)
    _model = tf.keras.models.load_model(_MODEL_PATH, compile=False)

    # Warm-up pass: forces TensorFlow to build the full computation graph so
    # that subsequent Grad-CAM gradient tapes are stable and repeatable.
    _model(tf.zeros((1, 224, 224, 3)), training=False)
    logger.info("Model loaded and warmed up successfully.")

    return _model


def predict(img_array: np.ndarray) -> tuple[np.ndarray, tf.keras.Model]:
    """
    Run a single forward pass and return class probabilities.

    Args:
        img_array: Float or uint8 array of shape ``(1, 224, 224, 3)``.
                   Values can be in [0, 255]; the function does not apply
                   additional preprocessing beyond dtype casting.

    Returns:
        A ``(predictions, model)`` tuple where:

        - ``predictions`` — NumPy array of shape ``(1, num_classes)`` with
          softmax probabilities.
        - ``model`` — The loaded ``tf.keras.Model`` (useful for Grad-CAM).
    """
    model = load_model()

    # Cast to float32 explicitly; models compiled without mixed precision
    # will error on int or float64 inputs.
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    predictions = model(img_tensor, training=False)

    return predictions.numpy(), model