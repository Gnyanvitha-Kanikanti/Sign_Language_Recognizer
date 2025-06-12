# 🤟 Sign Language Recognition System

A real-time Sign Language Recognition system that uses computer vision to identify hand gestures and classify them into corresponding alphabets. This project helps bridge the communication gap for individuals with speech impairments using AI.

## 🚀 Features

- 🔍 Real-time hand detection using `cvzone.HandTrackingModule`
- 🧠 Deep learning classification model built with TensorFlow/Keras
- 🎯 Accurate gesture recognition and classification of sign language alphabets
- 💡 Live webcam integration with OpenCV
- 🖼️ Cropping and preprocessing of hand regions before prediction
- ✨ Custom overlay display of recognized signs

## 🛠️ Tech Stack

- **Language:** Python
- **Model Type:** CNN (trained with Teachable Machine and custom dataset)
- **Hardware:** Works with standard webcam
- **Libraries Used:**
  - `OpenCV` for real-time video processing
  - `cvzone` for easy hand detection
  - `TensorFlow` / `Keras` for training the classification model
  - `NumPy` and `Pandas` for data handling
  - `Matplotlib` for visualizations

## 📁 Project Structure

📦 Sign-Language-Recognition/

├── 📁 model/

│ ├── keras_model.h5 # Trained deep learning model

│ └── labels.txt # Labels corresponding to model output

├── 📄 DataCollection.py # Script to capture and label new gesture data

├── 📄 test.py # Real-time testing and prediction script

├── 📄 README.md # Project documentation


## 🧪 How it Works

1. A hand is detected from the webcam feed using cvzone's `HandDetector`.
2. The region of interest (ROI) is extracted and resized.
3. The image is preprocessed (resized, normalized, centered for any size).
4. The trained CNN model predicts the sign/gesture.
5. The predicted character is displayed in a bounding box over the hand.
