import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def load_data(data_dir):
    images = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(data_dir, file_name)
            image = cv2.imread(image_path)
            # Skalowanie obrazów do ustalonego rozmiaru (np. 224x224)
            image = cv2.resize(image, (224, 224))
            images.append(image)

            # Automatyczne przypisanie etykiety na podstawie nazwy pliku
            label = get_label_from_file_name(file_name)
            labels.append(label)
    return np.array(images), np.array(labels)

def get_label_from_file_name(file_name):
    if 'good_squat' in file_name:
        return 1  # Poprawny przysiad
    else:
        return 0  # Niepoprawny przysiad

# Wczytanie danych treningowych
train_images, train_labels = load_data('train')

# Normalizacja danych
train_images = train_images / 255.0

# Definicja modelu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Definicja callback'u ModelCheckpoint
checkpoint_callback = callbacks.ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Trenowanie modelu
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint_callback])
