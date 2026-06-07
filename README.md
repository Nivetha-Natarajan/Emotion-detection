# Emotion-detection
Emotion Detection Based Music Recommendation System 🎵😊
Project Overview

The Emotion Detection Based Music Recommendation System is an AI-powered application that detects a user's facial emotion using a webcam and recommends music based on the detected emotion.

The system uses Computer Vision and Deep Learning techniques to analyze facial expressions in real-time and redirect users to Spotify with songs that match their current mood.

**Features**

Real-time face detection using OpenCV
Facial emotion recognition using a trained CNN model
Detects the following emotions:
Happy
Sad
Angry
Fear
Surprise
Disgust
Neutral
Music recommendation based on detected emotion
Spotify integration for mood-based song suggestions
Interactive webcam interface

**Technologies Used**

Programming Language
Python
Libraries
OpenCV
NumPy
TensorFlow / Keras
WebBrowser
Tools
Visual Studio Code
Git & GitHub

**Project Structure**
Emotion/
│
├── emotion.py
├── emotion_model.h5
├── README.md
├── .gitignore
└── requirements.txt

**Working Process**

The webcam captures the user's face.
OpenCV detects the face region.
The detected face is preprocessed and resized to 48×48 pixels.
The trained CNN model predicts the user's emotion.
The emotion is displayed on the screen.
When the user presses the C key, Spotify opens with songs related to the detected emotion.
The user can press Q to close the application.
Emotion to Music Mapping
Emotion	Music Category
Happy	Happy Songs
Sad	Emotional Songs
Angry	Workout Songs
Fear	Calm Songs
Surprise	Party Songs
Disgust	Dark Pop Songs
Neutral	Chill Songs
