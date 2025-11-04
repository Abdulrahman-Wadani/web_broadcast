import base64
import random
import cv2
import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return send_from_directory('static', 'broadcaster.html')

@socketio.on('frame')
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

    
    prediction_text = random.choice([
        "Prediction: Cat",
        "Prediction: Dog",
        "Prediction: Bird"
    ])
    # Replace with actual model prediction 

    socketio.emit('result', prediction_text)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

