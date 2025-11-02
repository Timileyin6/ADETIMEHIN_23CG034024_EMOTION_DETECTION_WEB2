import os
import io
import base64
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g
import numpy as np
import cv2


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "db_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_PATH = os.path.join(APP_ROOT, "users.db")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haar cascades (bundled with opencv-python)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

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

# --- Simple heuristic emotion detector ---
def detect_emotion_from_image(img_bgr):
    """Return one of: 'happy', 'surprised', 'neutral', 'sad/other'"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return "no_face_detected", None

    # pick largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_color = img_bgr[y:y+h, x:x+w]

    # smile detection
    smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=20)
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=8)

    # heuristic rules
    if len(smiles) > 0:
        emotion = "happy"
    elif len(eyes) >= 2 and h/w > 0.9:
        # relatively tall face + open eyes -> maybe surprised
        emotion = "surprised"
    else:
        # fallback guess
        emotion = "neutral"

    # cropped face for storing
    face_bgr = face_roi_color.copy()
    return emotion, face_bgr

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

    result, face_crop = detect_emotion_from_image(img_bgr)

    ts = datetime.utcnow().isoformat()
    filename = f"{name.replace(' ', '_')}_{ts.replace(':','-')}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # if face crop exists save that; else save full image
    if face_crop is not None:
        cv2.imwrite(save_path, face_crop)
    else:
        cv2.imwrite(save_path, img_bgr)

    # store record
    db = get_db()
    db.execute("INSERT INTO records (name, email, timestamp, image_path, result) VALUES (?, ?, ?, ?, ?)",
               (name, email, ts, save_path, result))
    db.commit()

    return jsonify({"status":"ok", "result": result, "image_saved": save_path})

@app.route("/records", methods=["GET"])
def records():
    db = get_db()
    rows = db.execute("SELECT id, name, email, timestamp, image_path, result FROM records ORDER BY id DESC LIMIT 100").fetchall()
    recs = [dict(r) for r in rows]
    return jsonify(recs)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
