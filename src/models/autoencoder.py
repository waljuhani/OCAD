
import tensorflow as tf
from tensorflow.keras import layers, models


LATENT_DIM = 64  # Latent space dimension
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
def build_encoder(input_shape=IMG_SHAPE):
    """
    Builds an encoder that reduces the channel dimensions from 256 -> 128 -> 64.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = models.Model(inputs, encoded, name="Encoder")
    return encoder, tf.keras.backend.int_shape(encoded)
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
def build_critic(latent_dim):
    """
    Builds the critic model to process flattened latent vectors of a defined size.

    Parameters:
        - latent_dim (int): Dimensionality of the latent vector expected by the critic.
    
    Returns:
        - critic (Model): The Keras Model representing the critic.
    """
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    alpha = layers.Dense(1, activation='sigmoid')(x)  # Interpolation coefficient
    return models.Model(inputs, alpha, name="Critic")
def build_decoder(shape_before_flatten):
    """
    Builds a decoder that mirrors the encoder structure by increasing channels from 64 -> 128 -> 256.
    """
    latent_input = layers.Input(shape=shape_before_flatten[1:])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(latent_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    # decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = models.Model(latent_input, decoded, name="Decoder")
    return decoder
# ----- Gaussian Anomaly Classifier -----
class GaussianAnomalyClassifier(layers.Layer):
    def __init__(self, latent_dim=LATENT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.mu = self.add_weight(shape=(latent_dim,), initializer="zeros", trainable=True)
        self.sigma = self.add_weight(shape=(latent_dim,), initializer="ones", trainable=True)

    def call(self, z):
        dist = (z - self.mu) ** 2 / (self.sigma ** 2)
        return tf.reduce_mean(dist, axis=-1)

def build_autoencoder():
    # Build encoder, decoder, anomaly classifier, and critic
    encoder, shape_before_flatten = build_encoder(IMG_SHAPE)
    decoder = build_decoder(shape_before_flatten)
    anomaly_classifier = GaussianAnomalyClassifier(LATENT_DIM)
    critic = build_critic(latent_dim=30976)  # Pass the latent_dim to ensure critic expects correct input size

    # Define Inputs
    x1 = layers.Input(shape=IMG_SHAPE)
    x2 = layers.Input(shape=IMG_SHAPE)

    # Encode x1 and x2
    z1 = encoder(x1)
    z2 = encoder(x2)

    # Debugging shapes (this will only show at model creation time)
    # print("Shape of z1:", z1.shape)
    # print("Shape of z2:", z2.shape)

    # Use Lambda to broadcast alpha to the shape of z1 (fixing the KerasTensor issue)
    alpha_broadcast = layers.Lambda(lambda z: tf.random.uniform(shape=tf.shape(z), minval=0, maxval=1))(z1)

    # Ensure 1 - alpha_broadcast has the same shape as alpha_broadcast
    one_minus_alpha = layers.Lambda(lambda alpha: 1 - alpha)(alpha_broadcast)

    # Debugging alpha shapes
    # print("Shape of alpha_broadcast:", alpha_broadcast.shape)
    # print("Shape of one_minus_alpha:", one_minus_alpha.shape)

    # Interpolate in latent space between z1 and z2
    z_alpha = layers.Add()([
        layers.Multiply()([z1, alpha_broadcast]),
        layers.Multiply()([z2, one_minus_alpha])
    ])

    # Decode latent representations (z1 and interpolated z_alpha)
    x1_hat = decoder(z1)
    x_alpha_hat = decoder(z_alpha)

    # Critic output for regularization (on interpolated latent space)
    critic_out = critic(layers.Flatten()(z_alpha))  # Flatten before passing to critic

    # Apply the Gaussian Anomaly Classifier to the latent representations
    anomaly_score_z1 = anomaly_classifier(z1)
    anomaly_score_z2 = anomaly_classifier(z2)

    # Define the complete model
    autoencoder_model = models.Model(
        inputs=[x1, x2],
        outputs=[x1_hat, critic_out, anomaly_score_z1, anomaly_score_z2],
        name="Autoencoder"
    )

    # Attach individual components for modular access
    autoencoder_model.encoder = encoder
    autoencoder_model.decoder = decoder
    autoencoder_model.anomaly_classifier = anomaly_classifier

    return autoencoder_model, encoder, decoder, anomaly_classifier, critic
