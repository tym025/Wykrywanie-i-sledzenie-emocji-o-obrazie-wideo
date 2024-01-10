# Import necessary libraries
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from classes_2 import KalmanFilter, Person
import keras.backend as K

# Define a mapping of emotion labels to human-readable names
label_2_emotion = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Function to prepare data for face vectorization
def preparing_data_for_face_vectorization(face_image):
    # Resize, convert to grayscale, and normalize the face image
    resized_face = cv2.resize(face_image, (95, 95))
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    face_img_array = img_to_array(gray_face)
    face_img_array_norm = face_img_array.astype('float32') / 255.0
    face_img_array_norm = np.expand_dims(face_img_array_norm, axis=0)
    return face_img_array_norm

# Function to prepare data for emotion detection
def preparing_data_for_emotion_detection(face_image):
    # Resize, convert to grayscale, and normalize the face image for emotion detection
    resized_face = cv2.resize(face_image, (48, 48))
    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
    face_img_array_norm = gray_face / 255
    reshaped_input = face_img_array_norm.reshape((1, 48, 48, 1))
    return reshaped_input

# Paths to the pre-trained models
checkpoint_path = 'siamese_epoch_69.h5'
siamese_model = load_model(checkpoint_path, compile=False)
sequential_layer = siamese_model.get_layer("sequential")
vectorizer_model = Model(inputs=siamese_model.input[0], outputs=sequential_layer.get_output_at(0))

model_path = 'model.h5'
model = load_model(model_path)

# Load the Haarcascades face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kalman Filter parameters
dt = 1.0 / 3600
F = np.array([[1, dt, 0, 0], [0, 1, dt, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = np.array([[0.05, 0.05, 0.0, 0.0], [0.05, 0.05, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
R = np.array([[0.5, 0], [0, 0.5]])

# Lists to store persons
persons = []

# OpenCV video capture from the default camera
cap = cv2.VideoCapture(0)

# Main loop for face tracking with Kalman Filter
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Iterate over detected faces
    for face in faces:
        # Extract the coordinates for measurement in Kalman Filter
        measurement_x = face[0] + 0.5 * face[2]
        measurement_y = face[1] + 0.5 * face[3]

        # If there are no persons being tracked, create a new person
        if len(persons) == 0:
            # Crop the face and prepare data for face vectorization
            croped_face = frame[face[1]:face[1]+face[2], face[0]:face[0]+face[3]]
            face_img = preparing_data_for_face_vectorization(croped_face)
            vectorized_face = vectorizer_model.predict(face_img)

            # Initialize Kalman Filter and create a new person
            kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
            new_person = Person(H, kf, measurement_x, measurement_y, vectorized_face)

            # Update Kalman Filter with measurement and emotion prediction
            new_person.kalman.update([measurement_x, measurement_y])
            pred = np.dot(H, new_person.kalman.predict())[0]
            new_person.x = measurement_x
            new_person.y = measurement_y
            new_person.kalman.update([measurement_x, measurement_y])

            # Prepare data for emotion detection
            face_img_for_emotion = preparing_data_for_emotion_detection(croped_face)
            predicted_label = np.argmax(model.predict(face_img_for_emotion), axis=1)
            new_person.emotions.append(label_2_emotion[predicted_label[0]])

            # Add the new person to the list
            persons.append(new_person)
        else:
            # Iterate over existing persons to check for matching persons
            for person in persons:
                if person.check(measurement_x, measurement_y):
                    # If a matching person is found, update Kalman Filter and emotions
                    croped_face = frame[face[1]:face[1]+face[2], face[0]:face[0]+face[3]]
                    face_img = preparing_data_for_face_vectorization(croped_face)
                    vectorized_face = vectorizer_model.predict(face_img)

                    if person.check_face(vectorized_face):
                        pred = np.dot(H, person.kalman.predict())[0]
                        if len(pred) > 1:
                            person.x = pred[0]
                            person.y = pred[1]
                        else:
                            person.x = measurement_x
                            person.y = measurement_y
                        person.kalman.update([measurement_x, measurement_y])

                        croped_face = frame[int(person.y):int(person.y+face[2]), int(person.x):int(person.x+face[3])]
                        face_img_for_emotion = preparing_data_for_emotion_detection(croped_face)
                        predicted_label = np.argmax(model.predict(face_img_for_emotion), axis=1)
                        person.emotions.append(label_2_emotion[predicted_label[0]])
                        break
            else:
                # If no matching person is found, create a new person
                croped_face = frame[face[1]:face[1]+face[2], face[0]:face[0]+face[3]]
                face_img = preparing_data_for_face_vectorization(croped_face)
                vectorized_face = vectorizer_model.predict(face_img)

                kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
                new_person = Person(H, kf, measurement_x, measurement_y, vectorized_face)

                new_person.kalman.update([measurement_x, measurement_y])
                pred = np.dot(H, new_person.kalman.predict())[0]
                new_person.x = measurement_x
                new_person.y = measurement_y
                new_person.kalman.update([measurement_x, measurement_y])

                face_img_for_emotion = preparing_data_for_emotion_detection(croped_face)
                predicted_label = np.argmax(model.predict(face_img_for_emotion), axis=1)
                new_person.emotions.append(label_2_emotion[predicted_label[0]])
                persons.append(new_person)

    # Display the frame with face tracking
    cv2.imshow('Face Tracking with Kalman Filter', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save data to a JSON file and break the loop
        persons_data = []
        for i, person in enumerate(persons):
            print(person.emotions, type(person.emotions))
            data = {
                "id": i,
                "emotions": person.emotions,
                "face": person.face.tolist(),
                "localization": [person.x, person.y],
            }
            persons_data.append(data)
        with open('data.json', 'w') as plik:
            json.dump(persons_data, plik)
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
