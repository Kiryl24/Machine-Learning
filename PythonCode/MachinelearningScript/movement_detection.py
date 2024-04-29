import cv2
from mediapipe.python.solutions import pose

# Inicjalizacja modelu MediaPipe Pose
pose_detector = pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def check_squat(previous_keypoints, current_keypoints):
    # Sprawdź, czy liczba punktów kluczowych się zgadza
    if len(previous_keypoints) != len(current_keypoints):
        return False

    # Sprawdź, czy którykolwiek punkt kluczowy zmienił swoje położenie
    for prev_point, curr_point in zip(previous_keypoints, current_keypoints):
        prev_x, prev_y, _, _ = prev_point
        curr_x, curr_y, _, _ = curr_point

        # Jeśli którykolwiek punkt kluczowy zmienił swoje położenie o więcej niż 0.2, zwróć True
        if abs(curr_x - prev_x) > 0.3 or abs(curr_y - prev_y) > 0.3:
            return True

    return False

# Pobierz obraz z kamery (zakładając, że masz dostęp do kamery)
cap = cv2.VideoCapture(0)

# Wczytaj pierwszą klatkę
ret, frame = cap.read()
if not ret:
    print("Błąd podczas wczytywania klatki z kamery.")
    exit()

# Zbierz punkty kluczowe na pierwszej klatce
results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
if results.pose_landmarks is not None:
    previous_keypoints = [(point.x, point.y, point.z, point.visibility) for point in results.pose_landmarks.landmark]
else:
    print("Nie wykryto postawy na pierwszej klatce.")
    exit()

while True:
    # Wczytaj kolejną klatkę
    ret, frame = cap.read()
    if not ret:
        print("Koniec strumienia wideo.")
        break

    # Zbierz punkty kluczowe na aktualnej klatce
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks is not None:
        current_keypoints = [(point.x, point.y, point.z, point.visibility) for point in results.pose_landmarks.landmark]

        # Sprawdź, czy punkty kluczowe zmieniły swoje położenie
        if check_squat(previous_keypoints, current_keypoints):
            print("Osoba wykonała przysiad!")
            # Tutaj możesz dodać dodatkowe działania, gdy przysiad zostanie wykryty

        # Zaktualizuj punkty kluczowe poprzedniej klatki
        previous_keypoints = current_keypoints

    # Wyświetl klatkę z zaznaczonymi punktami kluczowymi (opcjonalnie)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()
