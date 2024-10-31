

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.anomaly_detection_model import AnomalyDetectionModel
from utils.data_loader import create_data_generator
from models.autoencoder import build_autoencoder, build_encoder,build_decoder,build_critic



# Path to saved model
MODEL_PATH = '/content/drive/MyDrive/OCAD/models/best_anomaly_detection_model.keras'
TEST_DIR = '/content/drive/MyDrive/ISIC/Test/'

def main():
    # Load the saved model
    # Initialize the autoencoder model components
    autoencoder_model, encoder, decoder, anomaly_classifier, critic = build_autoencoder()
    
    # Initialize and compile the Anomaly Detection Model
    anomaly_detection_model = AnomalyDetectionModel(autoencoder_model=autoencoder_model, critic=critic)
    # anomaly_detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    anomaly_detection_model.load_weights(MODEL_PATH)
    print("Model loaded successfully.")

    # Set up test data generator
    test_generator = create_data_generator(
        directory=TEST_DIR,
        class_mode="binary",
        shuffle=False,
        classes=['Benign', 'Cancerous']
    )

    # Evaluate the model on the test set
    results = anomaly_detection_model.evaluate_model(test_generator)
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
