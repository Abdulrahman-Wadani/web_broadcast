import base64
import random
import cv2
import numpy as np
import io
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from gtts import gTTS
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
model.add(Dense(3, activation='softmax'))

print(os.getcwd())

#loading the best model
model.load_weights(r".\utils\DL_model\checkpoints\best_model.keras")

words = np.array(["أنا","هذا","اريد","شيء","هنا","الان","لا","في","ماذا","اخرس"])

mp_holistic = mp.solutions.holistic #holistics model

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

model_dict = pickle.load(open('./utils/letter-detection-model/py38model-best.p', 'rb'))
letter_model = model_dict['model']

# Initialize Mediapipe hands fo letter detection
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.3
word_frames_count=0

@app.route('/')
def index():
    return send_from_directory('static', 'broadcaster.html')

@socketio.on('Word_frame')
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
            
            prediction_text = words[np.argmax(res)]
            audio_buffer = io.BytesIO()

            tts = gTTS(text=prediction_text, lang="ar")
            tts.write_to_fp(audio_buffer)

            audio_bytes = audio_buffer.getvalue()
            b64_string = base64.b64encode(audio_bytes).decode('utf-8')
            data_uri = f"data:audio/mp3;base64,{b64_string}"
            
            sequence=[]
            socketio.emit('result', {"text": prediction_text,
                "url": data_uri})
    

labels_dict = {0: 'ع', 1: 'ال', 2: 'ا', 3: 'ب', 4: 'د', 5: 'ظ', 6: 'ض', 7: 'ف', 
8: 'ق', 9: 'غ', 10: 'ه', 11: 'ح', 12: 'ج', 13: 'ك', 14: 'خ', 
15: 'لا', 16: 'ل', 17: 'م', 18: 'ن', 19: 'ر', 20: 'ص', 21: 'س', 
22: 'ش', 23: 'ت', 24: 'ط', 25: 'ث', 26: 'ذ', 27: 'ة', 28: 'و', 
29: ' ', 30: 'ي', 31: 'ز'}


text_to_display = ""
previous_char = None
char_occurrences_counter = 0
predicted_character = None
required_occurrences = 10

data_aux = []
x_ = []
y_ = []


@socketio.on('Letter_frame')
def predict(data):

    global labels_dict
    global text_to_display
    global previous_char
    global char_occurrences_counter
    global predicted_character
    global required_occurrences
    
    global data_aux 
    global x_ 
    global y_ 
    
    
    
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

    try:
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                    
                # Prediction of the character
                prediction = letter_model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                
                data_aux = []
                x_ = []
                y_ = []

                if predicted_character is not None:
                    if predicted_character == previous_char:
                        char_occurrences_counter += 1
                    else:
                        previous_char = predicted_character
                        char_occurrences_counter = 0
                    
                    if char_occurrences_counter >= required_occurrences:
                        print(char_occurrences_counter)
                        char_occurrences_counter=0
                        text_to_display += predicted_character
                                         
        if predicted_character is not None:#return text after letter pred
            print(predicted_character)
            socketio.emit('result', {"text": text_to_display,
            "url": None})
            
    except Exception as e:#in case raised both hands, the program should produce an audio and reset the text
       print("===============================================\n",e,"\n==========================================================")
       try:
            audio_buffer = io.BytesIO()

            tts = gTTS(text=text_to_display, lang="ar")
            tts.write_to_fp(audio_buffer)

            audio_bytes = audio_buffer.getvalue()
            b64_string = base64.b64encode(audio_bytes).decode('utf-8')
            data_uri = f"data:audio/mp3;base64,{b64_string}"
            
            text_to_display = "";         
            socketio.emit('result', {"text": text_to_display,
                "url": data_uri})
       except Exception as e:
           print("===============Second Exception==========================\n",e,"\n==========================================================")
    
    

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)