import cv2
import os
from mediapipe.python.solutions import pose
from pytube import YouTube

# Inicjalizacja modelu MediaPipe do detekcji postury
pose_detector = pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def analyze_posture(image):
    # Konwersja kolorów BGR do RGB (MediaPipe wymaga obrazów w formacie RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detekcja postury za pomocą MediaPipe
    results = pose_detector.process(image_rgb)

    # Jeśli postura zostanie wykryta
    if results.pose_landmarks is not None:
        # Zwróć punkty kluczowe postury
        return results.pose_landmarks.landmark
    else:
        return None


def draw_keypoints(image, keypoints):
    image_with_keypoints = image.copy()
    image_height, image_width, _ = image.shape

    for keypoint in keypoints:
        keypoint_x = int(keypoint.x * image_width)
        keypoint_y = int(keypoint.y * image_height)
        cv2.circle(image_with_keypoints, (keypoint_x, keypoint_y), 5, (0, 255, 0), -1)

    return image_with_keypoints


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    frames = []
    success, image = cap.read()
    while success:
        frames.append(image)
        success, image = cap.read()

    cap.release()

    for i, frame in enumerate(frames):
        # Zapisz klatkę do analizy
        frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)

        # Analiza postury
        keypoints = analyze_posture(frame)

        if keypoints is not None:
            print("Analiza postury na klatce", frame_path)
            print("Punkty kluczowe postury:", keypoints)

            # Wizualizacja punktów kluczowych na klatce
            frame_with_keypoints = draw_keypoints(frame, keypoints)

            # Zapisz klatkę z wizualizacją punktów kluczowych
            frame_with_keypoints_path = os.path.join(output_dir, f"frame_{i}_with_keypoints.jpg")
            cv2.imwrite(frame_with_keypoints_path, frame_with_keypoints)


def crawl_videos(youtube_urls):
    training_data = []
    os.makedirs("train", exist_ok=True)

    for url in youtube_urls:
        yt = YouTube(url)
        video = yt.streams.filter(file_extension='mp4', res="360p").first()  # Wybierz wideo o rozdzielczości 360p
        video_path = video.download(output_path="videos", filename="video")
        extract_frames(video_path, "train")

        frames = os.listdir("train")

        for frame_name in frames:
            frame_path = os.path.join("train", frame_name)
            frame = cv2.imread(frame_path)

            # Analiza postury
            keypoints = analyze_posture(frame)

            if keypoints is not None:
                training_data.append((frame_path, keypoints))

    return training_data


if __name__ == "__main__":
    # Lista URL-i do filmów na YouTube
    youtube_urls = [
        "https://www.youtube.com/watch?v=xqvCmoLULNY"
    ]

    # Pobieranie danych treningowych
    training_data = crawl_videos(youtube_urls)

    # Przykładowe użycie danych treningowych
    for image_path, keypoints in training_data:
        print("Obraz:", image_path)
        print("Punkty kluczowe postury:", keypoints)
