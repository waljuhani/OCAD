
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SHAPE = (176, 176, 3)  # Input image shape

def self_attention_block(inputs, num_heads=4, key_dim=64):
    """
    Adds a self-attention mechanism to capture spatial dependencies within the feature map.
    
    Parameters:
        - inputs: Input tensor from the encoder layer.
        - num_heads: Number of attention heads.
        - key_dim: Dimensionality of the attention key.
        
    Returns:
        - x: Output tensor after applying self-attention and residual connection.
    """
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    x = layers.Add()([inputs, attention_output])
    x = layers.LayerNormalization()(x)
    return x

def build_encoder(input_shape=IMG_SHAPE):
    """
    Builds an encoder that reduces the channel dimensions from 256 -> 128 -> 64.
    
    Parameters:
        - input_shape (tuple): Shape of the input images.
    
    Returns:
        - encoder: Keras Model of the encoder.
        - shape_before_flatten: Shape of the encoded output before flattening.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = self_attention_block(x)  # Self-attention block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = models.Model(inputs, encoded, name="Encoder")
    return encoder, tf.keras.backend.int_shape(encoded)

def build_decoder(shape_before_flatten):
    """
    Builds a decoder that mirrors the encoder structure by increasing channels from 64 -> 128 -> 256.
    
    Parameters:
        - shape_before_flatten: Shape of the encoded input before flattening.
        
    Returns:
        - decoder: Keras Model of the decoder.
    """
    latent_input = layers.Input(shape=shape_before_flatten[1:])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(latent_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = models.Model(latent_input, decoded, name="Decoder")
    return decoder
