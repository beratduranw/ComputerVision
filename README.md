Hand Gesture Recognition with Mediapipe and Machine Learning

This project aims to capture hand gestures using the Mediapipe library, preprocess the hand landmarks, and classify the gestures using machine learning models (Random Forest Classifier and K-Nearest Neighbors). The goal is to recognize hand gestures in real time through a webcam feed.

Features

- Hand Detection: Uses Mediapipe to detect hand landmarks.
- Gesture Classification: Trains two classifiers (Random Forest and K-Nearest Neighbors) on captured hand landmarks data.
- Real-time Prediction: Predicts the gesture based on the current hand positions and displays the predicted gesture and its probability on the screen.
- Data Logging: Allows logging of hand landmarks and gestures to a CSV file for model training.

Dependencies

The following libraries are required to run the project:

- collections
- math
- csv
- cv2 (OpenCV)
- numpy
- mediapipe
- itertools
- sklearn
- joblib
- pandas
- warnings

To install the necessary libraries, you can run:

pip install opencv-python mediapipe scikit-learn joblib pandas

Project Structure

The main parts of the code include:

1. Data Logging: Captures and logs hand landmarks to a CSV file when save mode is enabled (press Enter to start logging).
2. Preprocessing: Preprocesses hand landmarks by normalizing and scaling them before passing them to the classifiers.
3. Training the Classifiers: Trains two machine learning classifiers on hand landmarks and stores the trained models for later use.
4. Real-time Prediction: Loads the trained Random Forest classifier and predicts the hand gesture in real time using the webcam feed.

How to Run

1. Data Collection:
   - Run the code, and press Enter (key 13) to start saving hand landmark data.
   - Ensure the camera is capturing hand gestures.
   - The captured hand landmark data is stored in data.csv.

2. Train the Model:
   - After collecting enough data, the train() function will split the data into training and testing sets, train the models (Random Forest and KNN), and print the accuracy of both models.
   - Trained models are saved as knn_model.joblib and rfc_model.joblib.

3. Real-time Gesture Recognition:
   - The code will start capturing video from the webcam and detecting hand gestures in real time.
   - Predictions are made using the Random Forest classifier and displayed on the screen, along with the confidence level.

4. Exit:
   - Press Esc to quit the real-time detection.

Important Functions

- shiftNormalized(): Normalizes the hand landmarks based on their bounding box and scales them to a fixed width and height.
- logCsv(): Logs the hand landmark data to the CSV file.
- train(): Trains the Random Forest and KNN classifiers on the logged data.
- CvFpsCalc: Calculates and displays the FPS for real-time performance tracking.
- calc_bounding_rect(): Calculates the bounding box for the detected hand landmarks.

Notes

- Ensure the webcam is connected and working properly.
- You can increase or decrease the number of trees in the Random Forest (n_estimators) or neighbors in KNN (n_neighbors) to experiment with model performance.
