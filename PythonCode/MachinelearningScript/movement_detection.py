import cv2
from mediapipe.python.solutions import pose
import mediapipe

def check_squat(previous_keypoints, current_keypoints):
    # Sprawdź, czy liczba punktów kluczowych się zgadza
    if len(previous_keypoints) != len(current_keypoints):
        return False

    # Sprawdź, czy którykolwiek punkt kluczowy zmienił swoje położenie
    for prev_point, curr_point in zip(previous_keypoints, current_keypoints):
        prev_x, prev_y, _, _ = prev_point
        curr_x, curr_y, _, _ = curr_point

        # Jeśli którykolwiek punkt kluczowy zmienił swoje położenie o więcej niż 0.2, zwróć True
        if abs(curr_x - prev_x) > 0.2 or abs(curr_y - prev_y) > 0.2:
            return True

    return False

def are_keypoints_available(results, required_points):
    """
    Sprawdza, czy wymagane punkty kluczowe są dostępne.
    """
    if results.pose_landmarks is None:
        return False

    landmarks = results.pose_landmarks.landmark
    for point in required_points:
        if landmarks[point].visibility < 0.5:  # Sprawdź, czy punkt jest widoczny
            return False
    return True


if __name__ == "__main__":
    print("Start")
    # Inicjalizacja modelu MediaPipe Pose
    pose_detector = pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    required_points = [
        pose.PoseLandmark.LEFT_HIP,
        pose.PoseLandmark.RIGHT_HIP,
        pose.PoseLandmark.LEFT_KNEE,
        pose.PoseLandmark.RIGHT_KNEE,
        pose.PoseLandmark.LEFT_ANKLE,
        pose.PoseLandmark.RIGHT_ANKLE,
        pose.PoseLandmark.LEFT_SHOULDER,
        pose.PoseLandmark.RIGHT_SHOULDER,
    ]

    # Pobierz obraz z kamery (zakładając, że masz dostęp do kamery)
    cap = cv2.VideoCapture(0)
    print("Strat0 .")


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
    mp_drawing = mediapipe.solutions.drawing_utils


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

            # Narysuj sylwetkę postaci oraz punkty charakterystyczne
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, pose.POSE_CONNECTIONS)

        # Wyświetl klatkę z zaznaczonymi punktami kluczowymi (opcjonalnie)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Zwolnij zasoby
    cap.release()
    cv2.destroyAllWindows()
