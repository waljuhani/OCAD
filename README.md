
# Attention-Enhanced Interpolated Gaussian Descriptors for One-Class Anomaly Detection in Medical Imaging

## Project Overview
This repository contains the implementation of our novel One-Class Anomaly Detection (OCAD) framework, which leverages Attention-Enhanced Interpolated Gaussian Descriptors to address the challenges in medical image anomaly detection. Designed for applications where data labeling is limited or expensive, such as in dermatology for identifying malignant lesions, our framework is trained exclusively on benign samples to identify deviations indicative of malignancies.
Attention-Based Autoencoder: Employs self-attention to capture critical regions within benign medical images, enhancing the model's sensitivity to subtle anomalies.
Gaussian Anomaly Scoring: Integrates Gaussian scoring in the latent space to distinguish between benign and malignant representations, reducing false positives.
Latent Space Interpolation: Interpolates latent vectors, allowing the model to generalize and capture nuanced deviations in complex, high-dimensional data.
![model strucure](https://github.com/user-attachments/assets/e658bdad-17bd-43e5-afbe-36d263e5002f)
## Sample of Results

![results](https://github.com/user-attachments/assets/0a4a2eaa-9aee-4f77-b05f-cff400c88568)

## Repository Structure
- **src/**: Contains all source code, including the model, data loader, and utility functions.
  - **data_loader.py**: Prepares the data generators for loading images.
  - **models/**: Defines model architecture components such as the encoder, decoder, critic, and anomaly detection model.
  - **train.py**: Main script to train the model.
  - **test.py**: Script to evaluate the model.
- **requirements.txt**: Lists all dependencies for the project.
- **README.md**: Project description and usage instructions.


## Getting Started

### Prerequisites
Make sure you have Python and `pip` installed. Use the following command to install dependencies:

```bash
pip install -r requirements.txt
```
### Dataset
The project uses the ISIC dataset for training and testing. Place the dataset in the following structure:

/path/to/OCAD/data/
```bash
├── Train
│   ├── Benign
│   └── Cancerous
└── Test
    ├── Benign
    └── Cancerous
```
### Training
To train the model, run:
```bash
python src/train.py
```
### Testing
To evaluate the model, run:
```bash
python src/test.py
```
### Acknowledgments
Thanks to the ISIC dataset providers and the community for dataset resources and insights on anomaly detection.
