import cv2
import numpy as np
import tensorflow as tf

from mediapipe.python.solutions import pose

# Inicjalizacja modelu MediaPipe do detekcji postury
pose_detector = pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Wczytanie trenowanego modelu
model = tf.keras.models.load_model('best_model.keras')  # Załóżmy, że najlepszy model został zapisany jako 'best_model.h5'

# Funkcja do analizy postury i klasyfikacji
def analyze_posture_and_classify(frame):
    # Konwersja kolorów BGR do RGB (MediaPipe wymaga obrazów w formacie RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detekcja postury za pomocą MediaPipe
    results = pose_detector.process(image_rgb)

    # Jeśli postura zostanie wykryta
    if results.pose_landmarks is not None:
        # Pobierz punkty kluczowe postury
        keypoints = results.pose_landmarks.landmark

        # Przekształć punkty kluczowe postury na wektor cech
        features = np.array([(keypoint.x, keypoint.y, keypoint.z, keypoint.visibility) for keypoint in keypoints]).flatten()

        # Przekształć wektor cech na wejście dla modelu
        input_data = np.expand_dims(features, axis=0)

        # Skalowanie obrazu do rozmiaru oczekiwanego przez model (224x224)
        frame_resized = cv2.resize(frame, (224, 224))

        # Przewidywanie klasy za pomocą modelu
        prediction = model.predict(frame_resized[np.newaxis, ...])

        # Wizualizacja punktów kluczowych postury na klatce
        for keypoint in keypoints:
            x = int(keypoint.x * frame.shape[1])
            y = int(keypoint.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Wyświetlenie przewidywanego labela
        label = "good squat" if prediction > 0.5 else "bad squat"
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analiza postury i klasyfikacja
    processed_frame = analyze_posture_and_classify(frame)

    # Wyświetlenie obrazu z kamerki
    cv2.imshow('Squat Classifier', processed_frame)

    # Wyjście z pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
