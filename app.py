import base64
import random
import cv2
import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import os
import time
import mediapipe as mp
from tqdm import tqdm
import pandas
import pickle
import random


import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

#falsk app config
app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")




# Recreate the model architecture before loading weights

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(20, activation='softmax'))

print(os.getcwd())

#loading the best model
model.load_weights(r".\DL_model\checkpoints\best_model.h5")

words = [
    'Skeleton',
    'Skull',
    'Spine',
    'Rib cage',
    'Respiratory system',
    'Trachea',
    'Lungs',
    'Inhalation - Exhalation',
    'Digestive system',
    'Face',
    'Pharynx',
    'Liver',
    'Pancreas',
    'Small intestine',
    'Large intestine',
    'Appendix',
    'Nervous system',
    'Heart',
    'Five senses',
    'Muscle'
]


mp_holistic = mp.solutions.holistic #holistics model
mp_drawing = mp.solutions.drawing_utils #drawing utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)   #convert image from bgr (noraml color spacing) to RGB
    image.flags.writeable = False                   #set Image to not writeable to save memory
    results = model.process(image)                  #detect and predict holistic using mediapipe
    image.flags.writeable = True                    #set Image back to writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)   #convert image from RGB to BGR (back to noraml)
    return results

def extract_featurs(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) #because it has visiblity feature unlike the others
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, right_hand])


@app.route('/')
def index():
    return send_from_directory('static', 'broadcaster.html')



#detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.3


@socketio.on('frame')
def predict(data):
    
    global words
    
    global sequence
    global sentence
    global predictions
    global threshold
    

    b64 = data.get("b64")
    if not b64:
        socketio.emit('result', "Error: No b64 data")
        return

    try:
        img_bytes = base64.b64decode(b64)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: 
            raise Exception("الصورة تالفة")
    except Exception as e:
        socketio.emit('result', f"Error: Bad image ({e})")
        return
    
    
    
    #1. Make detections
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print(tf.test.gpu_device_name())
        
        results = mediapipe_detection(frame, holistic)  
        print(results)
        
        #2. Prediction logic
        keypoints = extract_featurs(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        print(len(sequence))
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(words[np.argmax(res)])
            sequence = []
            socketio.emit('result',  words[np.argmax(res)])
            
    
        
    


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

