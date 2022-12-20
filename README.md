# Implementation of "Learning-from-Human-Feedback" - Project

From the paper "Learning Behaviors with Uncertain Human Feedback" with code found here :
https://github.com/hlhllhLearning-Behaviors-with-Uncertain-Human-Feedback

# Gesture feedback 

with TechVidvan Real-time Hand Gesture Recognition model found here : https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

## REQUIREMENTS 

pip install opencv-python

pip install mediapipe

pip install tensorflow

## HOW TO RUN 

Run the abluf_cartoon_gesture.py code,

Close the figure of the rat chasing to turn on the camera,

Sign your feedback : - "open hand" : Dog caught/is close to the mouse   
                     - "peace sign" : Dog is far from the mouse  
                     - "fist" : no feedback  
                     - "thumnbs up" :go to next state  

# Speech feedback

## REQUIREMENTS 

pip install speechrecognition

pip install portaudio

pip install pyaudio

## HOW TO RUN 

Run the abluf_cartoon_speech.py code,

Say your feedback : - "Good" : Dog caught/is close to the mouse   
                    - "Bad" : Dog is far from the mouse  
                    - "Neutral" : no feedback  
                    - "State" :go to next state   

#

### Tips : 
- Run the code from terminal (figures may not plot correctly with some IDEs)
- Create a data depository to save your training
- Don't forget to check the report !
