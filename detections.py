#import libraries

import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

#load up our trained model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

#import libraries

import cv2
import mediapipe as mp
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils #setup utitlities
mp_holistic = mp.solutions.holistic #load up our model

#load up webcam
cap = cv2.VideoCapture('explaining.mp4')



#initialize holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.7, min_tracking_confidence = 0.6) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #recolor feed
        image.flags.writeable = False #prevents copying image data
        results = holistic.process(image) #make detections
        #print(results.pose_landmarks) #print results

        ##Export Coordinates##
        try:
            # Extract the Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract the face landmarks
            face = results.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # concat the face and pose rows together
            row = pose_row + face_row

            ##MAKE DETECTIONS##
            x = pd.DataFrame([row]) #put the row data in a dataframe
            body_language_class = model.predict(x)[0] #predict and extract first value
            body_language_prob = model.predict_proba(x)[0] #get prediction of that class
            print(body_language_class, body_language_prob)

            ##SHOWING UP OUR RESULTS VISUALLY##

            #grab ear cordinates and put them in a numpy array and multiply by our screen size dimensions
            #make sure its in form of tuple which is recommended for opencv
            cords = tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y,
                                                )), [w, h]).astype(int))

            #render our predictions
            cv2.rectangle(image, (cords[0], cords[1]+30), (cords[0]+len(body_language_class)*32, cords[1]-30),
                          (245, 117, 16), -1)

            if body_language_class == 'BLM':
                cv2.putText(image, 'Black Lives Matter', cords,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, body_language_class, cords,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

            #Get Status Box
            #cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.rectangle(image, (0, 0), (180, 80), (245, 117, 16), cv2.FILLED)

            #Display class
            #cv2.putText(image, 'CLASS', (100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(image, body_language_class, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Probability
            cv2.putText(image, 'PROBABILITY', (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ######
            #cv2.putText(img, 'Jarvis: {}'.format(done), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)





        except:
            pass

        #Drawing specs
        mp_drawing.DrawingSpec(color = (0,255, 0), thickness = 2, circle_radius = 2)

        image.flags.writeable = False  # prevents copying image data
        #recolor image back to BGR for rendering with opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw facial landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (0,128,0), thickness = 1, circle_radius = 1),
                                  mp_drawing.DrawingSpec(color = (0,128,0), thickness = 1, circle_radius = 1))

        # draw right hand
        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  #mp_drawing.DrawingSpec(color = (255,0, 0), thickness = 2, circle_radius = 2),
                                  #mp_drawing.DrawingSpec(color = (0,0, 255), thickness = 2, circle_radius = 2))

        # draw left hand
        #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  #mp_drawing.DrawingSpec(color = (255,0, 0), thickness = 2, circle_radius = 2),
                                  #mp_drawing.DrawingSpec(color = (0,0, 255), thickness = 2, circle_radius = 2))


        # draw pose detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (0,128,0), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color = (238,130,238), thickness = 1, circle_radius = 2))



        cv2.imshow('Holistic Model Detections', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()





