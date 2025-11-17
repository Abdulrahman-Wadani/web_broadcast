import base64
import random
import cv2
import numpy as np
import io
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from gtts import gTTS

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return send_from_directory('static', 'broadcaster.html')

@socketio.on('Word_frame')
def predict(data):

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

    
    prediction_text = random.choice(["أنا","هذا","ُاريد","شيء","هنا","الان","لا","في","ماذا","اخرس"])

    audio_buffer = io.BytesIO()

    tts = gTTS(text=prediction_text, lang="ar")
    tts.write_to_fp(audio_buffer)

    audio_bytes = audio_buffer.getvalue()
    b64_string = base64.b64encode(audio_bytes).decode('utf-8')
    data_uri = f"data:audio/mp3;base64,{b64_string}"
    
    socketio.emit('result', {"text": prediction_text,
        "url": data_uri})
    

@socketio.on('Letter_frame')
def predict(data):

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

    
    prediction_text = random.choice(["أنا","هذا","ُاريد","شيء","هنا","الان","لا","في","ماذا","اخرس"])

    audio_buffer = io.BytesIO()

    tts = gTTS(text=prediction_text, lang="ar")
    tts.write_to_fp(audio_buffer)

    audio_bytes = audio_buffer.getvalue()
    b64_string = base64.b64encode(audio_bytes).decode('utf-8')
    data_uri = f"data:audio/mp3;base64,{b64_string}"
    
    socketio.emit('result', {"text": prediction_text,
        "url": data_uri})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

