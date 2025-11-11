import os
import io
import base64
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
import numpy as np
import cv2
from tensorflow.keras.models import load_model


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "db_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_PATH = os.path.join(APP_ROOT, "users.db")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haar cascades (bundled with opencv-python)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pre-trained emotion model
try:
    emotion_model = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
    model_loaded = True
    print("✓ Emotion model loaded successfully")
except Exception as e:
    print(f"⚠ Failed to load emotion model: {e}")
    model_loaded = False

# Emotion labels (7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Database helpers ---
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                timestamp TEXT,
                image_path TEXT,
                result TEXT
            )
        """)
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# --- Improved emotion detector using trained model ---
def detect_emotion_from_image(img_bgr):
    """
    Detect emotion using pre-trained Keras model.
    Returns: (emotion_label, face_crop, confidence_dict)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Enhanced face detection: use multiple scales
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,      # finer scale steps for better detection
        minNeighbors=5,         # stricter neighbor count
        minSize=(64, 64),       # require minimum face size
        maxSize=(400, 400)      # and maximum
    )
    
    if len(faces) == 0:
        return "no_face_detected", None, {}

    # Pick largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_color = img_bgr[y:y+h, x:x+w]

    # Preprocess for model: resize to match trained model input (64x64), grayscale
    face_resized = cv2.resize(face_roi_gray, (64, 64))
    
    # Apply histogram equalization to improve contrast
    face_equalized = cv2.equalizeHist(face_resized)
    
    # Normalize to [0, 1] range
    face_normalized = face_equalized.astype('float32') / 255.0
    
    # Add channel and batch dimensions for model input
    face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)

    # Predict emotion
    if model_loaded:
        try:
            predictions = emotion_model.predict(face_input, verbose=0)
            confidence_dict = {EMOTION_LABELS[i]: float(predictions[0][i]) for i in range(len(EMOTION_LABELS))}
            emotion_idx = np.argmax(predictions[0])
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(predictions[0][emotion_idx])
            
            # Only return prediction if confidence > 30%
            if confidence < 0.3:
                emotion = f"{emotion} (low confidence: {confidence:.1%})"
        except Exception as e:
            print(f"Model prediction error: {e}")
            emotion = "prediction_error"
            confidence_dict = {}
            confidence = 0.0
    else:
        # Fallback to rule-based if model not loaded
        emotion = "model_unavailable"
        confidence_dict = {}
        confidence = 0.0

    # Store cropped face
    face_bgr = face_roi_color.copy()
    
    return emotion, face_bgr, confidence_dict

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Accepts either a base64 image (from webcam) or a file upload."""
    name = request.form.get("name", "anonymous")
    email = request.form.get("email", "")
    image_b64 = request.form.get("imageBase64", None)

    if image_b64:
        # imageBase64 is like 'data:image/png;base64,AAAA...'
        header, encoded = image_b64.split(",", 1)
        data = base64.b64decode(encoded)
        img_arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    else:
        if 'file' not in request.files:
            return jsonify({"status":"error","message":"No file provided"}), 400
        file = request.files['file']
        img_bytes = file.read()
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"status":"error","message":"Failed to decode image"}), 400

    result, face_crop, confidence_dict = detect_emotion_from_image(img_bgr)

    ts = datetime.utcnow().isoformat()
    filename = f"{name.replace(' ', '_')}_{ts.replace(':','-')}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # if face crop exists save that; else save full image
    if face_crop is not None:
        cv2.imwrite(save_path, face_crop)
        # encode face crop as base64 data URL so frontend can preview immediately
        try:
            _, jpg = cv2.imencode('.jpg', face_crop)
            face_base64 = 'data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode('ascii')
        except Exception:
            face_base64 = None
    else:
        cv2.imwrite(save_path, img_bgr)
        face_base64 = None

    # store record
    db = get_db()
    db.execute("INSERT INTO records (name, email, timestamp, image_path, result) VALUES (?, ?, ?, ?, ?)",
               (name, email, ts, save_path, result))
    db.commit()

    return jsonify({
        "status": "ok", 
        "result": result, 
        "image_saved": save_path,
        "confidence": confidence_dict,  # include all emotion probabilities
        "face_base64": face_base64
    })

@app.route("/records", methods=["GET"])
def records():
    db = get_db()
    rows = db.execute("SELECT id, name, email, timestamp, image_path, result FROM records ORDER BY id DESC LIMIT 100").fetchall()
    recs = [dict(r) for r in rows]
    return jsonify(recs)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
