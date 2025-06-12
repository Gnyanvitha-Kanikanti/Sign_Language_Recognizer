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

- **Programming Language:** Python
- **Libraries Used:**
  - `OpenCV` for real-time video processing
  - `cvzone` for easy hand detection
  - `TensorFlow` / `Keras` for training the classification model
  - `NumPy` and `Pandas` for data handling
  - `Matplotlib` for visualizations

## 📁 Project Structure

📦 sign-language-recognition/
├── 📁 Dataset/ # Custom dataset of hand gestures
├── 📄 model.h5 # Trained Keras classification model
├── 📄 DataCollection.py # Module to create an custom hand dataset
├── 📄 test.py # Main app for real-time detection
├── 📄 README.md # You're here


## 🧪 How it Works

1. A hand is detected from the webcam feed using cvzone's `HandDetector`.
2. The region of interest (ROI) is extracted and resized.
3. The image is preprocessed (resized, normalized, centered for any size).
4. The trained CNN model predicts the sign/gesture.
5. The predicted character is displayed in a bounding box over the hand.
