## CNN Image Classification – Malaria Cell Detection 🦠🩸

Building a Convolutional Neural Network (CNN) model with and without Discrete Wavelet Transform (DWT) preprocessing to classify infected vs. uninfected malaria cell images. The project highlights dataset preparation, CNN architecture design, model training with different hyperparameters, and evaluation using multiple performance metrics.

---

📌 **Table of Contents**

- [Overview](#overview)  
- [Business Problem](#business-problem)  
- [Dataset](#dataset)  
- [Tools & Technologies](#tools--technologies)  
- [Project Structure](#project-structure)  
- [Data Cleaning & Preparation](#data-cleaning--preparation)  
- [Model Architecture](#model-architecture)  
- [Training & Evaluation](#training--evaluation)  
- [Results & Key Findings](#results--key-findings)  
- [Insights](#insights)  
- [Final Recommendations](#final-recommendations)  




## Overview

This project uses **Deep Learning (CNN)** to solve a binary classification problem of malaria cell images into Parasitized and Uninfected.

The workflow included:  
  
- Importing and organizing datasets from Google Drive  
- Preprocessing images (resizing, normalization)  
- Designing and training a CNN model  
- Comparing models with and without DWT-based feature extraction
- Evaluating accuracy and visualizing predictions  


## Business Problem

Malaria diagnosis using microscopes is time-consuming and error-prone. Automating this process with AI helps in:

- Reducing human error in early detection
- Speeding up diagnosis in healthcare centers
- Assisting doctors with reliable decision support 

**Objective:** Develop a robust CNN model capable of accurately detecting malaria-infected cells. 


## Dataset

- Malaria Cell Images Dataset (from NIH / Kaggle)

- ~27,500 labeled cell images

- Two categories: Parasitized and Uninfected

- Format: JPEG images with varied resolutions

**Directory Structure:**

/Malaria-Cell-Images

├── Parasitized/

├── Uninfected/




## Tools & Technologies

- **Programming:** Python  
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib, PyWavelets (for DWT) 
- **Environment:** Google Colab / Jupyter Notebook (GPU-accelerated)  
- **Version Control:** GitHub  



## Project Structure


├── data/                # Malaria dataset (Parasitized/Uninfected images)

├── models/              # Saved CNN models

├── scripts/             # Data preprocessing, DWT transform, training, evaluation

├── results/             # Confusion matrices & metric reports

├── malaria_cnn.ipynb    # CNN without DWT

├── malaria_dwt_cnn.ipynb # CNN with DWT preprocessing

└── README.md            # Project documentation



## Data Cleaning & Preparation

- Mounted dataset from Google Drive  
- Converted images into NumPy arrays
- Resized images to 128x128 pixels
- Normalized pixel values to [0,1]
- Applied Discrete Wavelet Transform (DWT) for frequency-based feature extraction in some experiments
- Split dataset into Train (80%) and Test (20%)



## Model Architecture

CNN Model built using **Keras Sequential API** with variations in convolution filters.  

- **Conv2D + ReLU Activation**  
- **MaxPooling2D**  
- **Dropout (to reduce overfitting)**  
- **Flatten layer**  
- **Dense (Fully Connected Layer)**  
- **Output Layer:** Sigmoid activation (binary classification)  



## Training & Evaluation

- **Optimizer:** Adam (learning rates: 0.0005 & 0.001)
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy, Precision, Recall, F1-score, MCC
- **Batch Size:** 32  
- **Epochs:** 50

**Evaluation was done using:** 
- Confusion Matrices
- Classification Metrics
- MCC (for balanced performance evaluation)



## Results & Key Findings

** Without DWT – lr=0.001, Variation (16,32,64)**
- Accuracy: 0.9612
- Precision: 0.9621 | Recall: 0.9612 | F1-score: 0.9612
- MCC: 0.9233

** With DWT – lr=0.0005, Variation (32,64,128) **
- Accuracy: 0.9565
- Precision: 0.9565 | Recall: 0.9565 | F1-score: 0.9565
- MCC: 0.9129

** With DWT – lr=0.0005, Variation (16,32,64) **
- Accuracy: 0.9661
- Precision: 0.9662 | Recall: 0.9661 | F1-score: 0.9661
- MCC: 0.9323

** With DWT – lr=0.001, Variation (32,64,128) **
- Accuracy: 0.9614
- Precision: 0.9616 | Recall: 0.9614 | F1-score: 0.9614
- MCC: 0.9229

** With DWT – lr=0.001, Variation (16,32,64) **
- Accuracy: 0.9541
- Precision: 0.9541 | Recall: 0.9541 | F1-score: 0.9541
- MCC: 0.9082


## 🔑 Insights

- CNN models with DWT preprocessing generally achieved higher accuracy & MCC, showing DWT helps in extracting key frequency features.

- The best performance was achieved with DWT + lr=0.0005 + (16,32,64) filters → Accuracy: 96.61%, MCC: 0.9323.

- Models without DWT also performed strongly, but slightly less consistent across variations.

## Final Recommendations

- Use DWT preprocessing for feature extraction to improve CNN robustness.
- Explore Transfer Learning (VGG16, ResNet) for further accuracy improvement.
- Deploy the best-performing model as a diagnostic aid for healthcare professionals.
- Extend project to multi-class classification (detecting multiple diseases from blood smear images). 


