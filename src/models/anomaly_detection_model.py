
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

@tf.keras.utils.register_keras_serializable()
class AnomalyDetectionModel(Model):
    """
    Anomaly Detection Model using autoencoder-based reconstruction loss, critic regularization, 
    and Gaussian anomaly scoring.

    Attributes:
        - autoencoder_model: Pre-trained autoencoder model.
        - critic: Critic model for latent space regularization.
        - threshold: Threshold value for anomaly detection.
    """
    def __init__(self, autoencoder_model, critic, **kwargs):
        super().__init__(**kwargs)
        self.autoencoder_model = autoencoder_model
        self.critic = critic
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.threshold = tf.Variable(0.5, trainable=False)  # Store threshold as a non-trainable variable

    def custom_reconstruction_loss(self, y_true, y_pred):
        """
        Computes reconstruction loss using MAE and SSIM for improved reconstruction quality.

        Parameters:
            - y_true: Original image batch.
            - y_pred: Reconstructed image batch.

        Returns:
            - Total reconstruction loss.
        """
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        return self.mae(y_true, y_pred) + ssim_loss

    def critic_loss(self, alpha_pred):
        """
        Critic loss to ensure balanced interpolation in latent space.

        Parameters:
            - alpha_pred: Critic prediction values.
        """
        return tf.reduce_mean(tf.square(alpha_pred - 0.5))

    def anomaly_loss(self, z, mu, sigma):
        """
        Gaussian anomaly loss for latent space regularization.

        Parameters:
            - z: Latent vector.
            - mu: Mean for anomaly scoring.
            - sigma: Standard deviation for anomaly scoring.
        """
        sigma = tf.maximum(sigma, 1e-5)
        return tf.reduce_mean(tf.square((z - mu) / sigma))

    def calculate_threshold(self, anomaly_loss_z1, anomaly_loss_z2):
        """
        Calculates and sets anomaly detection threshold based on mean anomaly scores.

        Parameters:
            - anomaly_loss_z1, anomaly_loss_z2: Anomaly loss scores for two latent representations.
        """
        threshold = (anomaly_loss_z1 + anomaly_loss_z2) / 2
        self.threshold.assign(threshold)
        print(f"Threshold updated: {threshold}")

    def train_step(self, data):
        """
        Custom training step with threshold update and gradient application.

        Parameters:
            - data: Input data tuple (x1).
        """
        x1 = data[0]
        x2 = x1  # Duplicate x1 to use as second input

        with tf.GradientTape() as tape:
            x1_hat, critic_pred, anomaly_score_z1, anomaly_score_z2 = self.autoencoder_model([x1, x2], training=True)
            z1 = self.autoencoder_model.encoder(x1)
            z2 = self.autoencoder_model.encoder(x2)

            recon_loss = self.custom_reconstruction_loss(x1, x1_hat)
            critic_loss = self.critic_loss(critic_pred)
            anomaly_loss_z1 = self.anomaly_loss(z1, self.autoencoder_model.anomaly_classifier.mu, self.autoencoder_model.anomaly_classifier.sigma)
            anomaly_loss_z2 = self.anomaly_loss(z2, self.autoencoder_model.anomaly_classifier.mu, self.autoencoder_model.anomaly_classifier.sigma)

            total_loss = 0.5 * recon_loss + 0.1 * critic_loss + anomaly_loss_z1 + anomaly_loss_z2
            self.calculate_threshold(anomaly_loss_z1, anomaly_loss_z2)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": total_loss}

    def evaluate_model(self, test_generator):
        """
        Evaluates the model on test data, calculating accuracy, precision, recall, and F1-score.

        Parameters:
            - test_generator: Data generator for test data.

        Returns:
            - results (dict): Evaluation metrics.
        """
        x_test, true_labels = next(iter(test_generator))
        x2 = x_test

        x_test_hat, _, _, _ = self.autoencoder_model([x_test, x2], training=False)
        mse = np.mean(np.square(x_test - x_test_hat), axis=(1, 2, 3))
        predicted_labels = (mse > self.threshold.numpy()).astype(int)

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        results = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
        print(f"Evaluation results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        return f1
