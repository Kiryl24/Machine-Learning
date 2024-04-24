import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


# Definicja ścieżki do danych treningowych
train_data_dir = 'squat_data'

# Parametry modelu
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)
batch_size = 32
epochs = 100

# Tworzenie modelu
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Kompilacja modelu
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Przygotowanie danych treningowych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,  # losowe obroty obrazu o kąt z zakresu 1-90 stopni
    horizontal_flip=True)  # losowe odbicie lustrzane obrazu

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Trenowanie modelu
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs)

# Zapisanie wytrenowanego modelu
model.save('squat_model.keras')
