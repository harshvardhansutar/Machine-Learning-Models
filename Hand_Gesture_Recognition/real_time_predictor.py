import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("P:\Machine_Learning_Models\Hand_Gesture_Recognition\models\hand_gesture_model.h5")
labels = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
          '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (64, 64))
    roi = roi.reshape(1, 64, 64, 1) / 255.0
    return roi

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    roi = frame[y1:y2, x1:x2]
    processed = preprocess_frame(roi)
    
    prediction = model.predict(processed, verbose=0)
    label = labels[np.argmax(prediction)]

    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
