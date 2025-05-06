from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp

app = Flask(__name__)
model = load_model('mobilenetv2_arabic_sign_model.h5')

arabic_labels = [
        'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah',
        'Heh', 'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf',
        'Reh', 'Sad', 'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal',
        'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_min, y_min = w, h
                    x_max = y_max = 0
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    margin = 20
                    x_min = max(x_min - margin, 0)
                    y_min = max(y_min - margin, 0)
                    x_max = min(x_max + margin, w)
                    y_max = min(y_max + margin, h)

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size == 0:
                        continue
                    hand_img = cv2.resize(hand_img, (224, 224))
                    image = img_to_array(hand_img)
                    image = np.expand_dims(image, axis=0)
                    image = preprocess_input(image)
                    
                    prediction = model.predict(image)[0]
                    idx = np.argmax(prediction)
                    name = arabic_labels[idx]  # atau tambahkan latin jika perlu
                    label = name

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    label_data = [
    ("Alef", "Alif", "ا", "Alef_30.jpg"),
    ("Beh", "Bāʾ", "ب", "Beh_180.jpg"),
    ("Teh", "Tāʾ", "ت", "Teh_240.jpg"),
    ("Theh", "Thāʾ", "ث", "Theh_302.jpg"),
    ("Jeem", "Jīm", "ج", "Jeem_198.jpg"),
    ("Hah", "Ḥāʾ", "ح", "Hah_20.jpg"),
    ("Khah", "Khāʾ", "خ", "Khah_189.jpg"),
    ("Dal", "Dāl", "د", "Dal_91.jpg"),
    ("Thal", "Dhāl", "ذ", "Thal_76.jpg"),
    ("Reh", "Rāʾ", "ر", "Reh_199.jpg"),
    ("Zain", "Zayn", "ز", "Zain_49.jpg"),
    ("Seen", "Sīn", "س", "Seen_180.jpg"),
    ("Sheen", "Shīn", "ش", "Sheen_276.jpg"),
    ("Sad", "Ṣād", "ص", "Sad_196.jpg"),
    ("Dad", "Ḍād", "ض", "Dad_70.jpg"),
    ("Tah", "Ṭāʾ", "ط", "Tah_171.jpg"),
    ("Zah", "Ẓāʾ", "ظ", "Zah_219.jpg"),
    ("Ain", "ʿAyn", "ع", "Ain_183.jpg"),
    ("Ghain", "Ghayn", "غ", "Ghain_186.jpg"),
    ("Feh", "Fāʾ", "ف", "Feh_168.jpg"),
    ("Qaf", "Qāf", "ق", "Qaf_144.jpg"),
    ("Kaf", "Kāf", "ك", "Kaf_35.jpg"),
    ("Lam", "Lām", "ل", "Lam_69.jpg"),
    ("Meem", "Mīm", "م", "Meem_113.jpg"),
    ("Noon", "Nūn", "ن", "Noon_142.jpg"),
    ("Heh", "Hāʾ", "ه", "Heh_15.jpg"),
    ("Waw", "Wāw", "و", "Waw_26.jpg"),
    ("Yeh", "Yāʾ", "ي", "Yeh_235.jpg"),
    ("Teh_Marbuta", "Tāʾ Marbūṭah", "ة", "Teh_Marbuta_215.jpg"),
    ("Laa", "Lāʾ", "لا", "Laa_12.jpg"),
    ("Al", "ʾĀl", "ال", "Al_259.jpg")
    ]
    return render_template('web.html', labels=label_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
