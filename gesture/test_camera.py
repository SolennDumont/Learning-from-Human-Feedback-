import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

img_background = cv2.imread('grass.jpg')
img_dog = cv2.imread('dog_w.png')
img_rat = cv2.imread('rat.png')
img_background = cv2.cvtColor(img_background, cv2.COLOR_BGR2RGB)
img_dog = cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)
img_rat = cv2.cvtColor(img_rat, cv2.COLOR_BGR2RGB)

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

def gesture_recognition():

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        user_input_temp = []
        
        while True :

                # Read each frame from the webcam
                _, frame = cap.read()
                x, y, c = frame.shape

                # Flip the frame vertically
                frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get hand landmark prediction
                result = hands.process(framergb)

                className = ''

                # post process the result
                if result.multi_hand_landmarks :
                        landmarks = []
                        for handslms in result.multi_hand_landmarks:
                                for lm in handslms.landmark:
                                        # print(id, lm)
                                        lmx = int(lm.x * x)
                                        lmy = int(lm.y * y)

                                        landmarks.append([lmx, lmy])

                                # Drawing landmarks on frames
                                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                                # Predict gesture
                                prediction = model.predict([landmarks])
                                # print(prediction)
                                classID = np.argmax(prediction)
                                className = classNames[classID]

                # show the prediction on the frame
                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                # Show the final output
                cv2.imshow("Output", frame)
                print(className)

                if className == "live long" : # Good feedback : main ouverte
                        user_input_temp.append(0)
                elif className == "peace":  # Bad feedback : signe de paix
                        user_input_temp.append(1)       
                elif className == "fist":  # Neutral feedback : poing fermé
                        user_input_temp.append(2)       
                elif className == "thumbs up":  # Next step
                        user_input_temp.append(3)       
                
                if len(user_input_temp) == 10 :
                        break
                if cv2.waitKey(1) == ord('q'):
                        break

        user_input = np.argmax([user_input_temp.count(0), user_input_temp.count(1), user_input_temp.count(2), user_input_temp.count(3), user_input_temp.count(4)])

        cap.release()
        cv2.destroyAllWindows()

        return user_input

user_input = gesture_recognition()
print(user_input)

