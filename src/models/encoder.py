
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SHAPE = (176, 176, 3)

def self_attention_block(inputs, num_heads=4, key_dim=64):
    """
    Adds a self-attention mechanism to capture spatial dependencies within the feature map.
    
    Parameters:
        - inputs: Input tensor from the encoder layer.
        - num_heads: Number of attention heads.
        - key_dim: Dimensionality of the attention key.
    """
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    x = layers.Add()([inputs, attention_output])
    x = layers.LayerNormalization()(x)
    return x

def build_encoder_with_self_attention(input_shape=IMG_SHAPE):
    """
    Builds an encoder with self-attention to capture global spatial dependencies.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Self-attention block
    x = self_attention_block(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = models.Model(inputs, encoded, name="Encoder_with_SelfAttention")
    return encoder, tf.keras.backend.int_shape(encoded)
