# ============================================================================
# إعداد البيئة (يجب أن يكون أولاً)
# ============================================================================
import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # استخدام المعالج فقط
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# ============================================================================
# استيراد المكتبات
# ============================================================================
import base64
import io
import cv2
import numpy as np
import pickle
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from gtts import gTTS
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from threading import Lock

# ============================================================================
# إعداد التطبيق
# ============================================================================
app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=20_000_000,
    async_mode='threading'  # استخدام threading للتعامل مع عدة عملاء
)

# ============================================================================
# الثوابت والإعدادات
# ============================================================================
class Config:
    """إعدادات النظام"""
    # قائمة الكلمات المدعومة
    WORDS = np.array([
        "أنا", "هذا", "اريد", "شيء", "هنا", 
        "الان", "لا", "في", "ماذا", "اخرس"
    ])
    
    
    
    # عدد الإطارات المطلوبة
    WORD_SEQUENCE_LENGTH = 30
    LETTER_REQUIRED_OCCURRENCES = 5
    
    # مسارات الملفات
    WORD_MODEL_PATH = r".\utils\DL_model\checkpoints\best_model.keras"
    LETTER_MODEL_PATH = './utils/letter-detection-model/py38model-best.p'

# ============================================================================
# تهيئة نماذج الذكاء الاصطناعي
# ============================================================================
class ModelManager:
    """إدارة نماذج التعلم العميق - مشتركة بين جميع العملاء"""
    
    def __init__(self):
        print("="*60)
        print("🚀 بدء تهيئة نظام كشف لغة الإشارة")
        print("="*60)
        
        self.word_model = self._build_word_model()
        self.letter_model = self._load_letter_model()
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        
        # قفل للتحكم في الوصول المتزامن للنماذج
        self.word_model_lock = Lock()
        self.letter_model_lock = Lock()
        
        print("="*60)
        print("✨ النظام جاهز للاستخدام!")
        print("="*60)
    
    def _build_word_model(self):
        """بناء وتحميل نموذج الكلمات"""
        print("📦 تحميل نموذج كشف الكلمات...")
        
        model = Sequential([
            LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 1662)),
            LSTM(128, return_sequences=True, activation='tanh'),
            LSTM(64, return_sequences=False, activation='tanh'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.load_weights(Config.WORD_MODEL_PATH)
        
        # تسخين النموذج
        print("🔥 تسخين نموذج الكلمات...")
        dummy = np.zeros((1, 30, 1662), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        print("✅ نموذج الكلمات جاهز!\n")
        
        return model
    
    def _load_letter_model(self):
        """تحميل نموذج الحروف"""
        print("📦 تحميل نموذج كشف الحروف...")
        
        with open(Config.LETTER_MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
        
        model = model_dict['model']
        
        # تسخين النموذج
        print("🔥 تسخين نموذج الحروف...")
        try:
            dummy = np.zeros((1, 42), dtype=np.float32)
            _ = model.predict(dummy)
            print("✅ نموذج الحروف جاهز!\n")
        except Exception as e:
            print(f"⚠️ النموذج محمّل (تم تخطي التسخين: {e})\n")
        
        return model
    
    def create_hands_detector(self):
        """إنشاء كاشف يدين جديد لكل عميل"""
        return self.mp_hands.Hands(
            static_image_mode=True, 
            min_detection_confidence=0.3
        )

# ============================================================================
# معالجة الصور والفيديو
# ============================================================================
class ImageProcessor:
    """معالجة الصور واستخراج الميزات"""
    
    @staticmethod
    def decode_base64_image(b64_string):
        """فك تشفير صورة من Base64"""
        try:
            img_bytes = base64.b64decode(b64_string)
            frame = cv2.imdecode(
                np.frombuffer(img_bytes, np.uint8), 
                cv2.IMREAD_COLOR
            )
            if frame is None:
                raise Exception("الصورة تالفة")
            return frame
        except Exception as e:
            raise Exception(f"خطأ في فك تشفير الصورة: {e}")
    
    @staticmethod
    def mediapipe_detection(image, model):
        """كشف الجسم والوجه واليدين باستخدام MediaPipe"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results
    
    @staticmethod
    def extract_keypoints(results):
        """استخراج النقاط المفتاحية من النتائج"""
        pose = np.array([
            [res.x, res.y, res.z, res.visibility] 
            for res in results.pose_landmarks.landmark
        ]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        face = np.array([
            [res.x, res.y, res.z] 
            for res in results.face_landmarks.landmark
        ]).flatten() if results.face_landmarks else np.zeros(468*3)
        
        left_hand = np.array([
            [res.x, res.y, res.z] 
            for res in results.left_hand_landmarks.landmark
        ]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        
        right_hand = np.array([
            [res.x, res.y, res.z] 
            for res in results.right_hand_landmarks.landmark
        ]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, left_hand, right_hand])

# ============================================================================
# معالجة الصوت
# ============================================================================
class AudioProcessor:
    """إنشاء ملفات صوتية من النص"""
    
    @staticmethod
    def text_to_audio_base64(text, lang='ar'):
        """تحويل النص إلى صوت بصيغة Base64"""
        try:
            audio_buffer = io.BytesIO()
            tts = gTTS(text=text, lang=lang)
            tts.write_to_fp(audio_buffer)
            audio_bytes = audio_buffer.getvalue()
            b64_string = base64.b64encode(audio_bytes).decode('utf-8')
            return f"data:audio/mp3;base64,{b64_string}"
        except Exception as e:
            print(f"خطأ في تحويل النص إلى صوت: {e}")
            return None

# ============================================================================
# إدارة جلسات العملاء
# ============================================================================
class ClientSession:
    """حالة جلسة لكل عميل - معزولة تماماً"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        
        # للكلمات
        self.word_sequence = []
        
        # للحروف
        self.letter_text = ""
        self.letter_previous_char = None
        self.letter_char_counter = 0
        self.letter_data_aux = []
        self.letter_x = []
        self.letter_y = []
        
        # كاشف اليدين الخاص بهذا العميل
        self.hands_detector = None
        
        print(f"✅ تم إنشاء جلسة جديدة: {session_id}")
    
    def initialize_hands_detector(self):
        """تهيئة كاشف اليدين لهذا العميل"""
        if self.hands_detector is None:
            self.hands_detector = models.create_hands_detector()
    
    def cleanup(self):
        """تنظيف موارد الجلسة"""
        if self.hands_detector:
            self.hands_detector.close()
        print(f"🧹 تم تنظيف الجلسة: {self.session_id}")

# قاموس لتخزين جلسات العملاء
client_sessions = {}
sessions_lock = Lock()

def get_or_create_session(session_id):
    """الحصول على جلسة موجودة أو إنشاء واحدة جديدة"""
    with sessions_lock:
        if session_id not in client_sessions:
            client_sessions[session_id] = ClientSession(session_id)
        return client_sessions[session_id]

def remove_session(session_id):
    """إزالة جلسة عميل"""
    with sessions_lock:
        if session_id in client_sessions:
            client_sessions[session_id].cleanup()
            del client_sessions[session_id]

# ============================================================================
# تهيئة النظام
# ============================================================================
models = ModelManager()

# ============================================================================
# المسارات (Routes)
# ============================================================================
@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return send_from_directory('templates', 'index.html')

@app.route('/translate')
def translate_page():
    """صفحة الترجمة المباشرة"""
    return send_from_directory('templates', 'translate.html')

@app.route('/train')
def train_hub():
    """مركز التدريب"""
    return send_from_directory('templates', 'train.html')

@app.route('/train/letters')
def train_letters():
    """تدريب الحروف"""
    return send_from_directory('templates', 'train_letters.html')

@app.route('/train/words')
def train_words():
    """تدريب الكلمات"""
    return send_from_directory('templates', 'train_words.html')

@app.route('/test')
def test_page():
    """صفحة الاختبار"""
    return send_from_directory('templates', 'test.html')

# ============================================================================
# معالجات SocketIO - الاتصال
# ============================================================================
@socketio.on('connect')
def handle_connect():
    """عند اتصال عميل جديد"""
    session_id = request.sid  # معرف فريد لكل عميل
    print(f"🔌 عميل جديد متصل: {session_id}")
    get_or_create_session(session_id)

@socketio.on('disconnect')
def handle_disconnect():
    """عند قطع اتصال عميل"""
    session_id = request.sid
    print(f"🔌 عميل قطع الاتصال: {session_id}")
    remove_session(session_id)

# ============================================================================
# معالجات SocketIO - الكلمات
# ============================================================================
@socketio.on('Word_frame')
def handle_word_frame(data):
    """معالجة إطار فيديو للكلمات - مع عزل العميل"""
    session_id = request.sid
    session = get_or_create_session(session_id)
    
    b64 = data.get("b64")
    if not b64:
        emit('result', "خطأ: لا توجد بيانات")
        return
    
    try:
        # فك تشفير الصورة
        frame = ImageProcessor.decode_base64_image(b64)
        
        # كشف النقاط المفتاحية
        with models.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            results = ImageProcessor.mediapipe_detection(frame, holistic)
            keypoints = ImageProcessor.extract_keypoints(results)
            
            # إضافة للتسلسل الخاص بهذا العميل
            session.word_sequence.append(keypoints)
            session.word_sequence = session.word_sequence[-Config.WORD_SEQUENCE_LENGTH:]
            
            # التنبؤ عند اكتمال التسلسل
            if len(session.word_sequence) == Config.WORD_SEQUENCE_LENGTH:
                # استخدام قفل لمنع التداخل في استخدام النموذج
                with models.word_model_lock:
                    prediction = models.word_model.predict(
                        np.expand_dims(session.word_sequence, axis=0),
                        verbose=0
                    )[0]
                
                predicted_word = Config.WORDS[np.argmax(prediction)]
                audio_url = AudioProcessor.text_to_audio_base64(predicted_word)
                
                # إعادة تعيين التسلسل
                session.word_sequence = []
                
                # إرسال النتيجة لهذا العميل فقط
                emit('result', {
                    "text": predicted_word,
                    "url": audio_url
                })
    
    except Exception as e:
        emit('result', f"خطأ: {str(e)}")

# ============================================================================
# معالجات SocketIO - الحروف
# ============================================================================
@socketio.on('Letter_frame')
def handle_letter_frame(data):
    
    # قائمة الحروف المدعومة
    letter_dict = {
        0: 'ع', 1: 'ال', 2: 'ا', 3: 'ب', 4: 'د', 5: 'ظ', 6: 'ض', 7: 'ف',
        8: 'ق', 9: 'غ', 10: 'ه', 11: 'ح', 12: 'ج', 13: 'ك', 14: 'خ',
        15: 'لا', 16: 'ل', 17: 'م', 18: 'ن', 19: 'ر', 20: 'ص', 21: 'س',
        22: 'ش', 23: 'ت', 24: 'ط', 25: 'ث', 26: 'ذ', 27: 'ة', 28: 'و',
        29: ' ', 30: 'ي', 31: 'ز'
    }
    
    """معالجة إطار فيديو للحروف - مع عزل العميل"""
    session_id = request.sid
    session = get_or_create_session(session_id)
    
    b64 = data.get("b64")
    if not b64:
        emit('result', "خطأ: لا توجد بيانات")
        return
    
    try:
        # تهيئة كاشف اليدين إذا لم يكن موجوداً
        session.initialize_hands_detector()
        
        frame = ImageProcessor.decode_base64_image(b64)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = session.hands_detector.process(frame_rgb)
        
        # التحقق من رفع يدين (إشارة لإنهاء الكلمة)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            if session.letter_text:
                audio_url = AudioProcessor.text_to_audio_base64(session.letter_text)
                emit('result', {
                    "text": "",
                    "url": audio_url
                })
                session.letter_text = ""
                session.letter_char_counter = 0
                session.letter_previous_char = None
            return
        
        # معالجة يد واحدة
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # استخراج الإحداثيات
                for landmark in hand_landmarks.landmark:
                    session.letter_x.append(landmark.x)
                    session.letter_y.append(landmark.y)
                
                # تطبيع البيانات
                for landmark in hand_landmarks.landmark:
                    session.letter_data_aux.append(landmark.x - min(session.letter_x))
                    session.letter_data_aux.append(landmark.y - min(session.letter_y))
                
                # التنبؤ مع قفل النموذج
                with models.letter_model_lock:
                    prediction = models.letter_model.predict([
                        np.asarray(session.letter_data_aux)
                    ])
                predicted_char = letter_dict[int(prediction[0])]
                
                # إعادة تعيين البيانات المؤقتة
                session.letter_data_aux = []
                session.letter_x = []
                session.letter_y = []
                
                # عد التكرارات
                if predicted_char == session.letter_previous_char:
                    session.letter_char_counter += 1
                else:
                    session.letter_previous_char = predicted_char
                    session.letter_char_counter = 0
                
                # إضافة الحرف عند الوصول للعدد المطلوب
                if session.letter_char_counter >= Config.LETTER_REQUIRED_OCCURRENCES:
                    session.letter_text += predicted_char
                    session.letter_char_counter = 0
        
        emit('result', {
            "text": session.letter_text,
            "url": None
        })
    
    except Exception as e:
        print(f"خطأ في معالجة الحرف: {e}")

# ============================================================================
# معالجات SocketIO - الاختبار
# ============================================================================
@socketio.on('Test_Letter')
def handle_test_letter(data):
    
    # قائمة الحروف المدعومة
    letter_dict = {
        0: 'ع', 1: 'ال', 2: 'أ', 3: 'ب', 4: 'د', 5: 'ظ', 6: 'ض', 7: 'ف',
        8: 'ق', 9: 'غ', 10: 'ه', 11: 'ح', 12: 'ج', 13: 'ك', 14: 'خ',
        15: 'لا', 16: 'ل', 17: 'م', 18: 'ن', 19: 'ر', 20: 'ص', 21: 'س',
        22: 'ش', 23: 'ت', 24: 'ط', 25: 'ث', 26: 'ذ', 27: 'ة', 28: 'و',
        29: ' ', 30: 'ي', 31: 'ز'
    }
    
    """اختبار الحروف - مع عزل العميل"""
    session_id = request.sid
    session = get_or_create_session(session_id)
    
    b64 = data.get("b64")
    target_char = data.get("target")
    
    if not b64:
        emit('test_response', "خطأ: لا توجد بيانات")
        return
    
    try:
        # تهيئة كاشف اليدين إذا لم يكن موجوداً
        session.initialize_hands_detector()
        
        frame = ImageProcessor.decode_base64_image(b64)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = session.hands_detector.process(frame_rgb)
        
        confidence_score = 0.0
        predicted_char = None
        
        if results.multi_hand_landmarks:
            temp_x = []
            temp_y = []
            temp_data = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    temp_x.append(landmark.x)
                    temp_y.append(landmark.y)
                
                for landmark in hand_landmarks.landmark:
                    temp_data.append(landmark.x - min(temp_x))
                    temp_data.append(landmark.y - min(temp_y))
                
                with models.letter_model_lock:
                    probabilities = models.letter_model.predict_proba([
                        np.asarray(temp_data)
                    ])
                    best_idx = np.argmax(probabilities[0])
                    confidence_score = probabilities[0][best_idx]
                    
                    prediction = models.letter_model.predict([np.asarray(temp_data)])
                    predicted_char = letter_dict[int(prediction[0])]
        
        # تحويل الثقة إلى نسبة مئوية مفهومة
        human_score = np.interp(confidence_score, [0.2, 0.8], [50, 100])
        human_score = round(min(100, max(0, human_score)), 1)
        
        if predicted_char == target_char:
            result = f"✅ صحيح! (الدقة: {human_score}%)"
        else:
            result = f"❌ خطأ! أنت أديت: {predicted_char} بدقة {human_score}%"
        
        emit('test_response', result)
    
    except Exception as e:
        emit('test_response', f"خطأ: {str(e)}")

@socketio.on('Test_Word')
def handle_test_word(data):
    """اختبار الكلمات - مع عزل العميل"""
    session_id = request.sid
    session = get_or_create_session(session_id)
    
    predicted_word = ""
    b64 = data.get("b64")
    target_word = data.get("target")
    
    if not b64:
        emit('result', "خطأ: لا توجد بيانات")
        return
    
    try:
        # فك تشفير الصورة
        frame = ImageProcessor.decode_base64_image(b64)
        
        # كشف النقاط المفتاحية
        with models.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            results = ImageProcessor.mediapipe_detection(frame, holistic)
            keypoints = ImageProcessor.extract_keypoints(results)
            
            # إضافة للتسلسل
            session.word_sequence.append(keypoints)
            session.word_sequence = session.word_sequence[-Config.WORD_SEQUENCE_LENGTH:]
            
            # التنبؤ عند اكتمال التسلسل
            if len(session.word_sequence) == Config.WORD_SEQUENCE_LENGTH:
                with models.word_model_lock:
                    prediction = models.word_model.predict(
                        np.expand_dims(session.word_sequence, axis=0),
                        verbose=0
                    )[0]
                
                predicted_idx = np.argmax(prediction)
                confidence = prediction[predicted_idx]
                predicted_word = Config.WORDS[predicted_idx]
                print(predicted_word)
                
                # إعادة تعيين التسلسل
                session.word_sequence = []
    
        if predicted_word == target_word:
            result = f"✅ صحيح! الدقة: {confidence:.0%}"
            emit('test_response', result)
        elif predicted_word != "":
            result = f"❌ خطأ! أنت أديت: {predicted_word}"
            emit('test_response', result)
    
    except Exception as e:
        emit('test_response', f"خطأ: {str(e)}")

# ============================================================================
# تشغيل التطبيق
# ============================================================================
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)