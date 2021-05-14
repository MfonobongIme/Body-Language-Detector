#import libraries

import cv2
import mediapipe as mp
import csv
import os
import numpy as np

class_name = 'Wakanda_forever'
mp_drawing = mp.solutions.drawing_utils #setup utitlities
mp_holistic = mp.solutions.holistic #load up our model

#body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml') #load cascade model for body detection

#load up webcam
cap = cv2.VideoCapture('wakanda_forever4.mp4')

#initialize holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.7, min_tracking_confidence = 0.6) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #recolor feed
        image.flags.writeable = False #prevents copying image data
        results = holistic.process(image) #make detections
        #print(results.pose_landmarks) #print results

        ##Export Coordinates for Wakanda_forever##
        try:
            # Extract the Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract the face landmarks
            face = results.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # concat the rows together
            row = pose_row + face_row

            # append the class name which is 'Wakanda_forever'
            row.insert(0, class_name)

            # Export to csv(append to cords.csv
            with open('cords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except:
            pass

        #Drawing specs
        mp_drawing.DrawingSpec(color = (0,255, 0), thickness = 2, circle_radius = 2)

        image.flags.writeable = False  # prevents copying image data
        #recolor image back to BGR for rendering with opencv
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw facial landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (42,42,165), thickness = 1, circle_radius = 1),
                                  mp_drawing.DrawingSpec(color = (42,42,165), thickness = 1, circle_radius = 1))

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
                                  mp_drawing.DrawingSpec(color = (0,0,255), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color = (255,255, 255), thickness = 2, circle_radius = 2))



        cv2.imshow('Holistic Model Detections', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




