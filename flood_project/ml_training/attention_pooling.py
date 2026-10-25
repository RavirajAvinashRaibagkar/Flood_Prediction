"""
attention_pooling.py  --  FloodSentinel AI
==========================================
Registered custom Keras layer that replaces the Lambda(tf.reduce_sum)
used previously. Using a proper @register_keras_serializable class lets
Keras serialize and deserialize the .keras file without needing any
unsafe_deserialization hacks.

BUG 2 FIX: Lambda layers cannot be saved/loaded cleanly in Keras 2/3.
  A registered subclass is the correct fix.

API compatibility:
  - TF 2.x (Keras 2 bundled):  keras.utils.register_keras_serializable
  - Keras 3 standalone:         keras.saving.register_keras_serializable
  This module detects which API is available and uses the right one.

Import this module before keras.models.load_model() in every consumer:
  from attention_pooling import AttentionPooling  # noqa: F401 (registers)
"""

import tensorflow as tf
from tensorflow import keras


def _get_register_fn():
    """Return the correct registration decorator for the installed Keras version."""
    # Keras 3 standalone has keras.saving.register_keras_serializable
    if hasattr(keras, 'saving') and hasattr(keras.saving, 'register_keras_serializable'):
        return keras.saving.register_keras_serializable
    # TF 2.x bundled Keras has keras.utils.register_keras_serializable
    if hasattr(keras.utils, 'register_keras_serializable'):
        return keras.utils.register_keras_serializable
    # Last resort: identity decorator (no registration; load via custom_objects)
    import warnings
    warnings.warn(
        "No keras register_keras_serializable found. "
        "AttentionPooling will not be auto-registered; "
        "pass custom_objects={'AttentionPooling': AttentionPooling} on load.",
        stacklevel=2,
    )
    return lambda package=None: (lambda cls: cls)


_register = _get_register_fn()


@_register(package="FloodSentinel")
class AttentionPooling(keras.layers.Layer):
    """
    Weighted sum-pooling over the time axis.

    Input : (batch, T, D)  -- context-weighted LSTM output
    Output: (batch, D)     -- sum across T; equivalent to tf.reduce_sum(x, axis=1)

    Registered in the Keras serialization registry so that .keras files
    round-trip without unsafe_deserialization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        # x: (batch, T, D)  ->  (batch, D)
        return tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        # (batch, T, D) -> (batch, D)
        return (input_shape[0], input_shape[2])
