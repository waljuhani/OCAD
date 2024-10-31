
import tensorflow as tf
from tensorflow.keras import layers, models

def build_decoder(shape_before_flatten):
    """
    Builds a decoder that mirrors the encoder structure by increasing channels from 64 -> 128 -> 256.

    Parameters:
        - shape_before_flatten (tuple): The shape before flattening, used as input to the decoder.
    
    Returns:
        - decoder (Model): The Keras Model representing the decoder.
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
