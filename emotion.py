import cv2
import numpy as np
import webbrowser
from tensorflow.keras.models import load_model

# =========================
# Load Model
# =========================
model = load_model("emotion_model.h5")

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

spotify_query = {
    'happy': 'happy songs',
    'sad': 'sad songs',
    'angry': 'angry workout songs',
    'surprise': 'party songs',
    'fear': 'calm songs',
    'disgust': 'dark pop songs',
    'neutral': 'chill songs'
}

# =========================
# Start Webcam
# =========================
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

current_emotion = None

print("Press C to open Spotify")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0

        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(
            roi_gray,
            verbose=0
        )

        emotion_index = np.argmax(prediction)
        current_emotion = emotion_labels[emotion_index]

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Emotion: {current_emotion}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "Press C = Spotify | Press Q = Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(30) & 0xFF

    # Press C
    if key == ord('c'):

        if current_emotion is not None:

            print("Detected Emotion:", current_emotion)

            search_url = (
                "https://open.spotify.com/search/"
                + spotify_query[current_emotion]
            )

            webbrowser.open(search_url)

    # Press Q
    elif key == ord('q'):
        print("Closing webcam...")
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()