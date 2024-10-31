
import tensorflow as tf
from tensorflow.keras import layers, models

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
