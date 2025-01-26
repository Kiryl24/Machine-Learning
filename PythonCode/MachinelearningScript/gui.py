from customtkinter import *
import cv2
from mediapipe.python.solutions import pose
import mediapipe
from PIL import Image, ImageTk
import threading

# Initialize the application
app = CTk()
app.geometry("360x780")
app.title("MyTrainer")

pose_detector = pose.Pose(static_image_mode=False,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)
running = False  # Flag to stop the camera feed
squat_count = 0  # Counter for squats


# Function to check if shoulders go down or up
def detect_movement(previous_avg_shoulder_y, current_avg_shoulder_y, body_height, going_down):
    """Detects if shoulders have moved down or up significantly."""
    if going_down:
        # Shoulders move down
        return (previous_avg_shoulder_y - current_avg_shoulder_y) >= (body_height * 0.2)
    else:
        # Shoulders move back up
        return (current_avg_shoulder_y - previous_avg_shoulder_y) >= (body_height * 0.2)


# Function to calculate the average y-position of the shoulders
def calculate_avg_shoulder_y(keypoints):
    left_shoulder_y = keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
    right_shoulder_y = keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]
    return (left_shoulder_y + right_shoulder_y) / 2


# Function to start the session
def start_session():
    global running
    running = True
    main_frame.pack_forget()  # Hide the main view
    session_frame.pack(fill="both", expand=True)  # Show the session view
    threading.Thread(target=display_camera_feed).start()


# Function to stop the session
def stop_session():
    global running
    running = False
    session_frame.pack_forget()  # Hide the session view
    main_frame.pack(fill="both", expand=True)  # Show the main view


# Function to display the camera feed in the GUI
def display_camera_feed():
    global squat_count, running

    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    required_points = [
        pose.PoseLandmark.LEFT_SHOULDER,
        pose.PoseLandmark.RIGHT_SHOULDER,
        pose.PoseLandmark.LEFT_HIP,
        pose.PoseLandmark.RIGHT_HIP,
    ]

    previous_avg_shoulder_y = None
    going_down = True  # Track the direction of movement (down or up)
    body_height = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = [
                (point.x, point.y, point.z, point.visibility)
                for point in results.pose_landmarks.landmark
            ]

            # Draw the skeleton on the frame
            mediapipe.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, pose.POSE_CONNECTIONS
            )

            # Check if required keypoints are available
            if all(results.pose_landmarks.landmark[point].visibility > 0.5 for point in required_points):
                status_label.configure(
                    text="GO DOWN!" if going_down else "GO UP!", text_color="green")

                # Calculate body height if not already calculated
                if body_height == 0:
                    left_hip_y = keypoints[pose.PoseLandmark.LEFT_HIP.value][1]
                    right_hip_y = keypoints[pose.PoseLandmark.RIGHT_HIP.value][1]
                    left_shoulder_y = keypoints[pose.PoseLandmark.LEFT_SHOULDER.value][1]
                    right_shoulder_y = keypoints[pose.PoseLandmark.RIGHT_SHOULDER.value][1]
                    avg_hip_y = (left_hip_y + right_hip_y) / 2
                    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                    body_height = avg_hip_y - avg_shoulder_y

                # Calculate average shoulder y-position
                current_avg_shoulder_y = calculate_avg_shoulder_y(keypoints)

                if previous_avg_shoulder_y is not None:
                    if going_down and detect_movement(previous_avg_shoulder_y, current_avg_shoulder_y, body_height, going_down):
                        going_down = False  # Switch to "going up" phase
                        status_label.configure(
                            text="GO UP!", text_color="orange")
                    elif not going_down and detect_movement(previous_avg_shoulder_y, current_avg_shoulder_y, body_height, going_down):
                        squat_count += 1
                        counter_label.configure(text=f"Squats: {squat_count}")
                        going_down = True  # Switch back to "going down" phase
                        status_label.configure(
                            text="GO DOWN!", text_color="green")

                previous_avg_shoulder_y = current_avg_shoulder_y
            else:
                status_label.configure(
                    text="User not detected. Please adjust your position.", text_color="red")

        else:
            status_label.configure(
                text="User not detected. Please adjust your position.", text_color="red")

        # Convert the frame to a format tkinter can display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (340, 400))
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        camera_label.configure(image=img)
        camera_label.image = img

        # Delay for smooth video feed
        app.update()

    cap.release()
    camera_label.configure(image="")  # Clear the camera feed


# Main view frame
main_frame = CTkFrame(app)
main_frame.pack(fill="both", expand=True)

header_frame = CTkFrame(main_frame, fg_color="transparent")
header_frame.pack(fill="x", padx=10, pady=(10, 20))

user_image = CTkLabel(header_frame, text="", width=40,
                      height=40, fg_color="red", corner_radius=20)
user_image.pack(side="left", padx=(10, 5))

greeting_label = CTkLabel(header_frame, text="Hi, user!", font=("Arial", 18))
greeting_label.pack(side="left")

card_frame = CTkFrame(main_frame, fg_color="#333333", corner_radius=15)
card_frame.pack(fill="x", padx=10, pady=20)

title_label = CTkLabel(card_frame, text="Squat", font=("Arial", 24, "bold"))
title_label.pack(anchor="center", pady=(10, 5))

description_label = CTkLabel(card_frame, text="Start squat training", font=(
    "Arial", 16), wraplength=300, justify="center")
description_label.pack(anchor="center", pady=(5, 10))

start_button = CTkButton(card_frame, text="Start session", width=200,
                         height=50, font=("Arial", 14), command=start_session)
start_button.pack(pady=(10, 10))

secondary_button = CTkButton(main_frame, text="Add new training", width=200, height=40, font=(
    "Arial", 14), fg_color="#555555", hover_color="#777777")
secondary_button.pack(pady=(10, 20))

# Session view frame
session_frame = CTkFrame(app)

camera_label = CTkLabel(session_frame, width=340, height=400)
camera_label.pack(pady=(10, 10))

status_label = CTkLabel(session_frame, text="User not detected.", font=(
    "Arial", 16), text_color="red")
status_label.pack(pady=(5, 10))

counter_label = CTkLabel(session_frame, text="Squats: 0", font=("Arial", 18))
counter_label.pack(pady=(10, 10))

stop_button = CTkButton(session_frame, text="Stop", width=200,
                        height=50, font=("Arial", 14), command=stop_session)
stop_button.pack(pady=(10, 10))

app.mainloop()
