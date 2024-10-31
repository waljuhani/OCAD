
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.anomaly_detection_model import AnomalyDetectionModel
from utils.data_loader import create_data_generator
from models.autoencoder import build_autoencoder, build_encoder,build_decoder,build_critic
from utils.callbacks import SaveBestF1Callback
import warnings

# Set TensorFlow logging to display only errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings if needed

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set TensorFlow logger level
tf.get_logger().setLevel('ERROR')
# Paths for training and testing datasets
TRAIN_DIR = '/content/drive/MyDrive/ISIC/Train/'
TEST_DIR = '/content/drive/MyDrive/ISIC/Test/'
SAVE_MODEL_PATH = '/content/drive/MyDrive/OCAD/models/best_anomaly_detection_model.keras'

# Training constants
IMG_HEIGHT, IMG_WIDTH = 176, 176
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

def main():
    # Create data generators for training and testing datasets
    train_generator = create_data_generator(
        directory=TRAIN_DIR,
        class_mode=None,  # One-class training for anomaly detection
        shuffle=True,
        augment=True
    )

    test_generator = create_data_generator(
        directory=TEST_DIR,
        class_mode="binary",
        shuffle=True,
        classes=['Benign', 'Cancerous']
    )

    # Initialize the autoencoder model components
    autoencoder_model, encoder, decoder, anomaly_classifier, critic = build_autoencoder()
    
    # Initialize and compile the Anomaly Detection Model
    anomaly_detection_model = AnomalyDetectionModel(autoencoder_model=autoencoder_model, critic=critic)
    anomaly_detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Set up callbacks for training
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    save_best_callback = SaveBestF1Callback(anomaly_detection_model, test_generator, save_path=SAVE_MODEL_PATH)

    callbacks = [early_stopping, lr_scheduler, save_best_callback]

    # Start training
    history = anomaly_detection_model.fit(
        train_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("Training completed. Best model saved to:", SAVE_MODEL_PATH)

if __name__ == "__main__":
    main()
