
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import Model, layers, callbacks

class SaveBestF1Callback(callbacks.Callback):
    def __init__(self, anomaly_detection_model, test_generator, save_path):
        super().__init__()
        self.anomaly_detection_model = anomaly_detection_model
        
        self.test_generator = test_generator
        self.best_f1 = 0.0  # Track the best F1 score
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        f1 = self.anomaly_detection_model.evaluate_model(self.test_generator)  # Evaluate the model
        if f1 > self.best_f1:  # Check if the F1 score improved
            print(f"F1 improved from {self.best_f1:.4f} to {f1:.4f}. Saving model...")
            self.best_f1 = f1  # Update the best F1 score
            self.anomaly_detection_model.save(self.save_path)  # Save the model
