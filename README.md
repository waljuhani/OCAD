
# Anovel Deep Learning Model for One-Class Anomaly Detection via Interpolated Gaussian Descriptor

## Project Overview
This project implements a novel deep learning-based one-class anomaly detection model using an interpolated Gaussian descriptor. The model focuses on identifying anomalous data in image datasets, with a particular application in medical imaging for detecting malignant lesions.
![model strucure](https://github.com/user-attachments/assets/e658bdad-17bd-43e5-afbe-36d263e5002f)
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
