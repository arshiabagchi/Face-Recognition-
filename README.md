# Facial-Recognition

## Overview
This project is a facial recognition system that allows users to create a dataset of labeled images, train a deep learning model, and recognize faces in real-time using a webcam. The dataset is stored in Google Drive, ensuring persistence across different sessions.

## Features
- Capture and store labeled facial images in Google Drive
- Train a Convolutional Neural Network (CNN) for facial recognition
- Use TensorFlow and Keras for model training
- Visualize dataset images and model predictions
- Predict and recognize faces in real-time

## Technologies Used
- Python (Google Colab)
- TensorFlow/Keras
- Scikit-learn
- Matplotlib
- Google Drive API

## Project Workflow
1. **Dataset Creation**
   - Mounts Google Drive and stores images in a specified directory.
   - Uses the webcam to capture labeled images and save them to the dataset.
   - Ensures each person has at least one image.

2. **Data Preprocessing**
   - Loads images, resizes them to 128x128, and normalizes pixel values.
   - Splits data into training and testing sets using `train_test_split`.

3. **Model Training**
   - Uses a CNN with convolutional and max-pooling layers.
   - Trains the model using categorical cross-entropy loss and Adam optimizer.
   - Evaluates the trained model on the test dataset.

4. **Face Recognition**
   - Captures a new image using the webcam.
   - Preprocesses the image and feeds it into the trained model.
   - Predicts and displays the recognized person.

## How to Use
1. **Run the Notebook in Google Colab**
   - Ensure that Google Drive is mounted properly.
   - Run the dataset creation script to capture images.

2. **Train the Model**
   - Execute the training script and monitor the accuracy.
   - If needed, adjust hyperparameters and retrain.

3. **Recognize Faces**
   - Capture a new image and run it through the trained model.
   - The model will predict and display the recognized person.
  
## Screenshots

   - Dataset
![image](https://github.com/user-attachments/assets/0c21a519-4017-487e-9f87-b0ffe7dc576c)

   - Prediction
![image](https://github.com/user-attachments/assets/aca1043b-0755-421a-8259-7933ed38d161)


## Acknowledgments
This project is inspired by facial recognition applications and aims to provide a basic yet effective implementation using deep learning techniques.

