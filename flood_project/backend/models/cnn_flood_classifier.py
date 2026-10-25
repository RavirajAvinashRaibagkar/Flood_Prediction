"""
CNN Flood Classifier Model -- Anti-Overfitting Edition
=======================================================
Input : 128x128x3 RGB satellite image (reduced from 224 for regularization)
Output: flood_probability (float 0-1)

Architecture choices (research-backed, 2024):
  * MobileNetV2 transfer learning backbone (ImageNet pretrained)
      -> Freezable feature extractor; avoids re-learning low-level filters
      -> Massively reduces effective parameter count vs. custom 3-layer CNN
  * Global Average Pooling instead of Flatten
      -> Reduces 4x4x1280 feature map to 1280-dim vector (structural regularizer)
  * L2 weight-decay on dense head
  * Double dropout (0.5 + 0.4) in head
  * BatchNormalization in head (stabilizes distribution shift between backbone & head)
  * Input reduced to 128x128 to limit memorization capacity on small datasets

Why MobileNetV2?
  * Pre-trained on 1.28M ImageNet images -> rich multi-scale spatial features
  * Lightweight depthwise-separable convolutions -> fewer params than VGG/ResNet
  * Satellite flood imagery (water reflectance, texture) re-uses ImageNet low-level filters
  * Consistently top performer in flood classification papers (IISTR 2024, MDPI 2024)

References:
  - "MobileNetV2 for Flood Detection" -- iistr.org 2024
  - "Transfer Learning Strategies for Satellite Flood Classification" -- MDPI 2024
"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Input resolution (smaller = less memorization capacity on small datasets)
INPUT_SHAPE = (128, 128, 3)

try:
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    def build_cnn_classifier(input_shape=INPUT_SHAPE, l2_factor=1e-4):
        """
        Build MobileNetV2-based flood classifier with anti-overfitting measures.

        Args:
            input_shape : (H, W, C) -- default (128, 128, 3)
            l2_factor   : L2 weight-decay strength on dense head (default 1e-4)

        Returns:
            Compiled Keras Model
        """
        # Backbone: MobileNetV2 (ImageNet pretrained, conv layers FROZEN initially)
        backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,   # remove ImageNet classifier head
            weights="imagenet",
            pooling=None,
        )
        backbone.trainable = False   # freeze all conv layers in Phase 1

        # Build model
        inputs = keras.Input(shape=input_shape, name="image_input")

        # Preprocess: MobileNetV2 expects pixels in [-1, 1]
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)

        # Feature extraction (frozen backbone)
        x = backbone(x, training=False)

        # Global Average Pooling -- structural regularizer (replaces Flatten)
        x = layers.GlobalAveragePooling2D(name="gap")(x)

        # BatchNorm before dense head -- stabilises backbone->head transition
        x = layers.BatchNormalization(name="bn_head")(x)

        # Dense classifier head with L2 regularization
        x = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_factor),
            name="dense_128",
        )(x)
        x = layers.Dropout(0.5, name="dropout_1")(x)

        x = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_factor),
            name="dense_64",
        )(x)
        x = layers.Dropout(0.4, name="dropout_2")(x)

        output = layers.Dense(1, activation="sigmoid", name="flood_prob")(x)

        model = keras.Model(inputs=inputs, outputs=output, name="FloodCNN_MobileNetV2")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        return model

    def unfreeze_backbone_top(model, num_layers_to_unfreeze=30):
        """
        Selectively unfreeze the top N layers of the MobileNetV2 backbone
        for fine-tuning in Phase 2. Keeps low-level edge/texture detectors frozen.

        Args:
            model                  : compiled Keras model with MobileNetV2 backbone
            num_layers_to_unfreeze : how many top backbone layers to thaw (default 30)
        """
        # MobileNetV2 layer name pattern
        backbone_name = f"mobilenetv2_1.00_{input_shape[0]}" if False else None
        for layer in model.layers:
            if hasattr(layer, 'layers'):   # it's a sub-model (the backbone)
                backbone = layer
                break
        else:
            print("   [WARN] Could not find backbone sub-model layer.")
            return

        backbone.trainable = True
        # Freeze all except the last N layers
        for layer in backbone.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        frozen = sum(1 for l in backbone.layers if not l.trainable)
        trainable = sum(1 for l in backbone.layers if l.trainable)
        print(f"   [UNFREEZE] Backbone: {frozen} frozen, {trainable} trainable (top {num_layers_to_unfreeze})")

    def load_trained_model():
        """Load the best available trained classifier checkpoint; never build a fresh model for inference."""
        model_dir = Path(__file__).resolve().parents[2] / "ml_training" / "saved_models"
        # Names must match what train_cnn_v2.py writes (cnn_v2_seed<N>.keras);
        # the older names are kept as fallbacks.
        candidates = sorted(model_dir.glob("cnn_v2_seed*.keras"), reverse=True) + [
            model_dir / "cnn_flood_classifier_phase3.keras",
            model_dir / "cnn_flood_classifier.h5",
        ]
        for checkpoint in candidates:
            if checkpoint.is_file():
                try:
                    model = keras.models.load_model(str(checkpoint), compile=False)
                    logger.info("Loaded flood classifier checkpoint %s", checkpoint.name)
                    return model
                except Exception as exc:
                    logger.warning("Could not load classifier checkpoint %s: %s", checkpoint, exc)
        logger.warning("No usable trained CNN checkpoint found in %s", model_dir)
        return None

    HAS_TENSORFLOW = True

except ImportError:
    HAS_TENSORFLOW = False

    def build_cnn_classifier(input_shape=INPUT_SHAPE, l2_factor=1e-4):
        """Stub -- TensorFlow not available."""
        return None

    def unfreeze_backbone_top(model, num_layers_to_unfreeze=30):
        """Stub -- TensorFlow not available."""
        pass

    def load_trained_model():
        return None


def predict_flood_probability(image_array: np.ndarray, model=None) -> dict:
    """
    Run flood classification on a preprocessed image.
    A missing checkpoint is reported explicitly rather than producing a random score.

    Args:
        image_array : np.ndarray of shape (H, W, 3) or (1, H, W, 3)
        model       : trained Keras model (optional)

    Returns:
        dict with flood_probability, confidence, model name, input_shape
    """
    if model is not None and HAS_TENSORFLOW:
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        # Determine expected shape from model if available
        exp_shape = getattr(model, "input_shape", None)
        if exp_shape and len(exp_shape) == 4 and exp_shape[1] is not None:
            target_size = (exp_shape[1], exp_shape[2])
        else:
            target_size = INPUT_SHAPE[:2]

        # Resize to expected shape if needed
        if image_array.shape[1:3] != target_size:
            import tensorflow as tf
            image_array = tf.image.resize(image_array, target_size).numpy()
        prediction = model.predict(image_array, verbose=0)
        prob = float(prediction[0][0])
    else:
        raise RuntimeError("Trained flood classifier is unavailable (TensorFlow or checkpoint missing).")

    return {
        "flood_probability": round(prob, 4),
        "confidence": round(abs(prob - 0.5) * 2, 4),   # distance from decision boundary
        "model": "FloodCNN-MobileNetV2-Transfer",
        "input_shape": f"{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}x{INPUT_SHAPE[2]}",
    }
