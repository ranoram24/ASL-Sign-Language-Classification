# ASL Alphabet Sign Language Recognition
**Authors:** Ran Uram, Shahar Lankry, and Daniel Geron

## Overview
This project implements an American Sign Language (ASL) alphabet recognition system utilizing Convolutional Neural Networks (CNNs). The objective of the system is to automatically detect and classify human hand gestures corresponding to the 26 letters of the English alphabet (A-Z) from static images. This repository contains the complete deep learning pipeline, from initial data preprocessing through to model architecture design and evaluation.

## Methodology & Technical Approach
To achieve high classification accuracy, we developed a structured machine learning pipeline focused on robust feature extraction and optimization.

### 1. Data Preprocessing
The model was trained on the Synthetic ASL Alphabet dataset, which consists of 26,000 images. To ensure the network could process the data efficiently and accurately:
*   **Standardization:** All images were resized to a uniform resolution to match the input layer requirements of our neural networks.
*   **Normalization:** Pixel intensity values were scaled to a standardized range. This step was crucial to accelerate the convergence of the gradient descent algorithm and improve overall model stability during training.

### 2. Model Architecture
We designed and evaluated multiple Convolutional Neural Network (CNN) architectures to find the optimal balance between computational efficiency and accuracy. The core architecture includes:
*   **Feature Extraction:** Sequential convolutional layers were utilized to extract spatial hierarchies of features from the raw images, progressively learning to identify edges, contours, and complex hand formations.
*   **Dimensionality Reduction:** Max-pooling layers were applied after convolutions to reduce spatial dimensions and computational load while retaining the most prominent extracted features.
*   **Classification:** Fully connected (Dense) layers were used at the end of the network, concluding with a softmax activation function to output a probability distribution across the 26 distinct alphabet classes.

### 3. Training & Optimization
To maximize performance and prevent the network from overfitting to the training data, we implemented several advanced training strategies:
*   **Callbacks:** We integrated Early Stopping to automatically halt training when validation metrics stopped improving. Alongside this, we utilized Learning Rate Reduction (ReduceLROnPlateau) to dynamically fine-tune the model's weights as it approached convergence.
*   **Regularization:** Dropout layers were incorporated throughout the architecture to ensure the model generalized well to unseen data rather than memorizing the synthetic training set.

## Performance & Results
The finalized CNN architecture demonstrated excellent reliability and robustness in classifying ASL hand signs. Upon final evaluation, the best-performing model achieved a validation and test accuracy of **95% - 98%**, confirming its high precision and viability for gesture classification tasks.

---

## Installation & Setup
To run the notebooks and evaluate the models locally, the required dataset must be downloaded and the local environment path configured.

### 1. Download the Dataset
The models are trained using the Synthetic ASL Alphabet dataset. You must download it before running the code:
[Download Dataset from Kaggle](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

### 2. Configure the Path
Extract the downloaded dataset on your local machine. Open the project notebook and locate the appropriate path declaration line near the top of the data import section:

```python
# Set this variable to your local dataset path
data_dir = "path/to/the/dataset/saved/locally"
