import sys, os
# ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import detect_emotion_from_image
import cv2
img = cv2.imread(os.path.join('db_images','_2025-11-02T18-13-41.837584.jpg'))
if img is None:
    print('Could not open test image')
else:
    emotion, face_crop, conf = detect_emotion_from_image(img)
    print('Emotion:', emotion)
    print('Confidences:', conf)
    if face_crop is not None:
        cv2.imwrite('tests/_last_face_crop.jpg', face_crop)
        print('Saved face crop to tests/_last_face_crop.jpg')
