import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from classes_2 import KalmanFilter, Person
import keras.backend as K

checkpoint_path = 'siamese_epoch_69.h5'
siamese_model = load_model(checkpoint_path, compile=False)

sequential_layer = siamese_model.get_layer("sequential")
vectorizer_model = Model(inputs=siamese_model.input[0], outputs=sequential_layer.get_output_at(0))

dt = 1.0 / 3600
F = np.array([[1, dt, 0, 0], [0, 1, dt, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q = np.array([[0.05, 0.05, 0.0, 0.0], [0.05, 0.05, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
R= np.array([[0.5, 0], [0, 0.5]])

label_2_emotion = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_path = 'emotion_model.h5'

model = load_model(model_path)
    
cap = cv2.VideoCapture(0)

true_location = []
pred_location = [] 
persons = []
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for face in faces:
        measurement_x = face[0] + 0.5 * face[2]
        measurement_y = face[1] + 0.5 * face[3]

        croped_face = frame[face[1]:face[1]+face[2] , face[0]:face[0]+face[3]]
        resized_croped_face = cv2.resize(croped_face, (95, 95))
        gray_croped_face = cv2.cvtColor(resized_croped_face, cv2.COLOR_BGR2GRAY)

        if len(persons) == 0:
            face_img = img_to_array(gray_croped_face)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            vectorized_face = vectorizer_model.predict(face_img)
            kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
            
            new_person = Person(H, kf, measurement_x, measurement_y, vectorized_face)
            pred = np.dot(H,  new_person.kalman.predict())[0]
            if len(pred)>1:
                new_person.x = pred[0]
                new_person.y = pred[1]
            else:
                new_person.x = 0.
                new_person.y = 0.
            new_person.kalman.update([measurement_x, measurement_y])
            
            resized_croped_2_face = cv2.resize(gray_croped_face, (48, 48))
            normalized_cropped_2_face = resized_croped_2_face / 255
            reshaped_input = normalized_cropped_2_face.reshape((1, 48, 48, 1))
            predicted_label = np.argmax(model.predict(reshaped_input), axis=1)
            new_person.emotions.append(label_2_emotion[predicted_label[0]])
            persons.append(new_person)
        else:
            for person in persons:
                if person.check(measurement_x, measurement_y):
                    face_img = img_to_array(gray_croped_face)
                    face_img = face_img.astype('float32') / 255.0
                    face_img = np.expand_dims(face_img, axis=0)
                    vectorized_face = vectorizer_model.predict(face_img)
                    if person.check_face(vectorized_face): 
                        pred = np.dot(H,  person.kalman.predict())[0]
                        if len(pred)>1:
                            person.x = pred[0]
                            person.y = pred[1]
                        else:
                            person.x = 0.
                            person.y = 0.
                        resized_croped_2_face = cv2.resize(gray_croped_face, (48, 48))
                        normalized_cropped_2_face = resized_croped_2_face / 255
                        reshaped_input = normalized_cropped_2_face.reshape((1, 48, 48, 1))
                        predicted_label = np.argmax(model.predict(reshaped_input), axis=1)
                        person.emotions.append(label_2_emotion[predicted_label[0]])
                        person.kalman.update([measurement_x, measurement_y])
                        break
            else:
                face_img = img_to_array(gray_croped_face)
                face_img = face_img.astype('float32') / 255.0
                face_img = np.expand_dims(face_img, axis=0)
                vectorized_face = vectorizer_model.predict(face_img)
            
                kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

                new_person = Person(H, kf, measurement_x, measurement_y, vectorized_face)
                new_person.kalman.update([measurement_x, measurement_y])
                pred = np.dot(H,  new_person.kalman.predict())[0]
                if len(pred)>1:
                    new_person.x = pred[0]
                    new_person.y = pred[1]
                else:
                    new_person.x = 0.
                    new_person.y = 0.

                resized_croped_2_face = cv2.resize(gray_croped_face, (48, 48))
                normalized_cropped_2_face = resized_croped_2_face / 255
                reshaped_input = normalized_cropped_2_face.reshape((1, 48, 48, 1))
                predicted_label = np.argmax(model.predict(reshaped_input), axis=1)
                new_person.emotions.append(label_2_emotion[predicted_label[0]])

                persons.append(new_person)

    cv2.imshow('Face Tracking with Kalman Filter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        persons_data = []  # Lista, do której będziemy dodawać dane każdej osoby
        for i, person in enumerate(persons):
            print(person.emotions, type(person.emotions))
            data = {
                "id": i,
                "emotions": person.emotions,
                "face": person.face.tolist(),
                "kalman": [person.x, person.y],
            }
            persons_data.append(data)
        with open('data.json', 'w') as plik:
            json.dump(persons_data, plik)
        break
cap.release()
cv2.destroyAllWindows()