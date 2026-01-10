# Human Action Recognition using CNN + LSTM

## 📌 Project Overview
This project implements a Deep Learning solution for **Human Action Recognition** in videos. It leverages **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for temporal sequence learning. The goal is to accurately classify various human actions from video sequences using the UCF50 dataset.

## 📂 Dataset
The project utilizes the **[UCF50 - Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF50.php)**, which is widely used for evaluating action recognition models.
- **50** Action Categories
- **Realistic videos** sourced from YouTube (unconstrained environments)
- **25** Groups of videos per category
- Handles variations in camera motion, object appearance, scale, and viewpoint.

## 🛠 Technologies & Libraries
The implementation is done in **Python** using the following key libraries:
- **TensorFlow / Keras**: Model building (CNNs, LSTMs, TimeDistributed layers).
- **OpenCV (cv2)**: Image and video processing (reading frames, resizing).
- **Matplotlib**: Data visualization and plotting training graphs.
- **MoviePy**: Video editing and handling.
- **Scikit-learn**: Data splitting (train/test split).

## 🧠 Approach & Architectures
This notebook explores and implements two deep learning architectures for video classification:

### 1. ConvLSTM Approach
- Utilizes **ConvLSTM2D** layers.
- Combines convolution operations directly inside the LSTM cells.
- Efficiently captures **spatiotemporal correlations** by maintaining the spatial structure of frames while processing temporal sequences.

### 2. LRCN (Long-term Recurrent Convolutional Network) Approach
- Combines a **TimeDistributed CNN** with **LSTM** layers.
- **Spatial Feature Extraction**: Each frame in the video sequence is passed through a CNN to extract features.
- **Temporal Modeling**: The sequence of extracted feature vectors is fed into LSTM layers to learn time-dependent patterns.
- A robust and standard approach for video analysis.

## 🚀 Workflow
The project follows a structured pipeline:
1.  **Data Visualization**: Loading and displaying samples from the dataset with labels.
2.  **Data Preprocessing**:
    - Resizing frames (e.g., to a fixed heigh/width).
    - Normalizing pixel values.
    - Converting videos to fixed-length sequences.
    - One-hot encoding of class labels.
3.  **Model Training**:
    - constructing both ConvLSTM and LRCN models.
    - Training with appropriate loss functions and optimizers.
    - Using callbacks like **EarlyStopping** to prevent overfitting.
4.  **Evaluation**:
    - Plotting **Accuracy** and **Loss** curves for training and validation sets.
    - Comparing model performance.
5.  **Inference**:
    - Testing the trained model on random YouTube videos or local video files.
    - Generating output videos with predicted action labels overlayed.

## 📦 How to Use
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```
2.  **Install Dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install tensorflow opencv-python matplotlib moviepy
    ```
3.  **Run the Notebook**:
    - Open `Human_Action_Recogntion_using_CNN_+_LSTM.ipynb` in Jupyter Notebook or Google Colab.
    - Run the cells sequentially to download the data, train the models, and see the results.

## 📊 Results
Detailed performance metrics and loss/accuracy plots are available within the notebook. The project concludes by demonstrating the model's ability to classify actions in unseen video clips.
