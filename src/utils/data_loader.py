
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_HEIGHT, IMG_WIDTH = 176, 176
BATCH_SIZE = 32

def create_data_generator(directory, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode="rgb", class_mode=None, shuffle=True, subset=None, classes=None, augment=False):
    """
    Creates a data generator for loading images from a specified directory with optional augmentation for training.

    Parameters:
        - directory (str): Path to the data directory.
        - target_size (tuple): The target image size for resizing (height, width).
        - batch_size (int): Number of images per batch.
        - color_mode (str): Color mode for images ('rgb' or 'grayscale').
        - class_mode (str or None): Classification mode ('categorical', 'binary', or None for unlabeled data).
        - shuffle (bool): Whether to shuffle the data.
        - subset (str or None): Subset of data to use ('training' or 'validation' for data split).
        - classes (list or None): List of class subdirectories (e.g., ['Benign', 'Cancerous']).
        - augment (bool): Whether to apply data augmentation (only for training).

    Returns:
        - generator (DirectoryIterator): A Keras data generator.
    """
    if augment:
        # Augmentations for training
        data_gen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        # No augmentations for testing
        data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    generator = data_gen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        subset=subset,
        classes=classes
    )

    return generator
