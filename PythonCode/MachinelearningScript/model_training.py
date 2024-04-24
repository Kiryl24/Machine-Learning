import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.models import Model
from keras.preprocessing.image import img_to_array

# Definicja szerokości i wysokości obrazu
img_width, img_height = 224, 224

# Funkcja do przetwarzania klatki obrazu
def process_frame(frame):
    frame = cv2.resize(frame, (img_width, img_height))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Tworzenie modelu
input_layer = Input(shape=(img_width, img_height, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Kompilacja modelu
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Przetwarzanie pliku MP4
video_file = 'example.mp4'
cap = cv2.VideoCapture(video_file)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for _ in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    prediction = model.predict(processed_frame)
    # Działania na predykcji (np. wyświetlenie, zapisanie itp.)

# Przetwarzanie pliku GIF
gif_file = 'example.gif'
gif_frames = cv2.VideoCapture(gif_file)
while True:
    ret, frame = gif_frames.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    prediction = model.predict(processed_frame)
    # Działania na predykcji (np. wyświetlenie, zapisanie itp.)

# Zakończenie odtwarzania plików
cap.release()
gif_frames.release()
cv2.destroyAllWindows()

model.save("trained_model.keras")

print("Model został pomyślnie zapisany do pliku trained_model.keras")