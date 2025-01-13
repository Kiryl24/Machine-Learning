import cv2
from mediapipe.python.solutions import pose
import mediapipe

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
        if abs(curr_x - prev_x) > 0.2 or abs(curr_y - prev_y) > 0.2:
            return True

    return False

# Pobierz obraz z kamery (zakładając, że masz dostęp do kamery)
cap = cv2.VideoCapture(0)
required_points = [
    pose.PoseLandmark.LEFT_HIP,
    pose.PoseLandmark.RIGHT_HIP,
    pose.PoseLandmark.LEFT_KNEE,
    pose.PoseLandmark.RIGHT_KNEE,
    #pose.PoseLandmark.LEFT_ANKLE,
    #pose.PoseLandmark.RIGHT_ANKLE,
    pose.PoseLandmark.LEFT_SHOULDER,
    pose.PoseLandmark.RIGHT_SHOULDER,
]

def are_keypoints_available(results, required_points):
    """
    Sprawdza, czy wymagane punkty kluczowe są dostępne.
    """
    if results.pose_landmarks is None:
        #print("False1.")
        return False

    landmarks = results.pose_landmarks.landmark
    for point in required_points:
        #print(landmarks[point].visibility)
        if landmarks[point].visibility < 0.4:  # Sprawdź, czy punkt jest widoczny
            #print("False2.")
            return False
    return True

def calculate_body_height(keypoints):
    """Oblicza wysokość ciała jako różnicę w osi y między ramionami a biodrami."""
    left_shoulder_y = keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
    right_shoulder_y = keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]
    left_hip_y = keypoints[pose.PoseLandmark.LEFT_HIP.value][1]
    right_hip_y = keypoints[pose.PoseLandmark.RIGHT_HIP.value][1]

    # Średnia wysokość ramion i bioder
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2

    return avg_hip_y - avg_shoulder_y

def check_shoulders_moved_down(previous_keypoints, current_keypoints, body_height):
    """Sprawdza, czy ramiona przesunęły się w dół o 1/3 wysokości ciała."""
    prev_left_shoulder_y = previous_keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
    prev_right_shoulder_y = previous_keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]

    curr_left_shoulder_y = current_keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
    curr_right_shoulder_y = current_keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]

    # Średnie przesunięcie ramion w dół
    prev_avg_shoulder_y = (prev_left_shoulder_y + prev_right_shoulder_y) / 2
    curr_avg_shoulder_y = (curr_left_shoulder_y + curr_right_shoulder_y) / 2

    # Sprawdzenie przesunięcia
    return (curr_avg_shoulder_y - prev_avg_shoulder_y) >= (body_height / 3)

def check_shoulders_moved_up(previous_keypoints,current_keypoints,body_height,difference):
    """Sprawdza, czy ramiona wróciły do pozycji poczatkowej z moznliwa roznica o warosc = body_height * difference."""
    prev_left_shoulder_y = previous_keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
    prev_right_shoulder_y = previous_keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]

    curr_left_shoulder_y = current_keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
    curr_right_shoulder_y = current_keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]

    # Średnie przesunięcie ramion w dół
    prev_avg_shoulder_y = (prev_left_shoulder_y + prev_right_shoulder_y) / 2
    curr_avg_shoulder_y = (curr_left_shoulder_y + curr_right_shoulder_y) / 2

    # Sprawdzenie przesunięcia
    return (curr_avg_shoulder_y - prev_avg_shoulder_y) <= (body_height * difference)

# Wczytaj pierwszą klatkę
ret, frame = cap.read()
if not ret:
    print("Błąd podczas wczytywania klatki z kamery.")
    exit()
import time
print("Sleeping..") # for debugging only
time.sleep(2)
# Zbierz punkty kluczowe na pierwszej klatce
results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
if results.pose_landmarks is not None:
    previous_keypoints = [(point.x, point.y, point.z, point.visibility) for point in results.pose_landmarks.landmark]
    body_height = calculate_body_height(previous_keypoints)
    print("Zebrano postawe poczatkowa")
else:
    print("Nie wykryto postawy na pierwszej klatce.")
    exit()

mp_drawing = mediapipe.solutions.drawing_utils
squat=0
i = 0
pozycja_poczatkowa = previous_keypoints
faza_przysiadu= False
while True:
    i = i + 1
    # Wczytaj kolejną klatkę
    ret, frame = cap.read()
    if not ret:
        print("Koniec strumienia wideo.")
        break

    # Zbierz punkty kluczowe na aktualnej klatce
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks is not None:
        current_keypoints = [(point.x, point.y, point.z, point.visibility) for point in results.pose_landmarks.landmark]
        if are_keypoints_available(results,required_points):
            if not faza_przysiadu:
                if check_shoulders_moved_down(pozycja_poczatkowa,current_keypoints,body_height):
                    squat=squat+1
                    print(f"Osoba wykonała {squat} przysiad! ")
                    faza_przysiadu=True

                    #time.sleep(0.3)
            elif check_shoulders_moved_up(pozycja_poczatkowa,current_keypoints,body_height,0.20):
                #print(f"Osoba skonczyla przysiad!")
                faza_przysiadu= False
        else:
            print(f"Nie wykryto calego ciala ! {i}")

        # Zaktualizuj punkty kluczowe poprzedniej klatki
        previous_keypoints = current_keypoints

        # Narysuj sylwetkę postaci oraz punkty charakterystyczne
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, pose.POSE_CONNECTIONS)
        time.sleep(0.05)

    # Wyświetl klatkę z zaznaczonymi punktami kluczowymi (opcjonalnie)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()

