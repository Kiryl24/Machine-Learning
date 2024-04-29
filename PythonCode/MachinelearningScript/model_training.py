import os
import yaml
import scipy
from keras_preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import tensorflow as tf

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_CLASSES = 6


def load_yaml_data(folder_path):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        folder_path,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=32,
        class_mode='binary'
    )
    images, labels = next(generator)
    return images, labels
# Przygotowanie danych treningowych
X_train, y_train = load_yaml_data("C:/Users/Ewa/Documents/GitHub/Machine-Learning/PythonCode/MachinelearningScript")

# Sprawdzenie kształtu danych treningowych
print("Kształt X_train:", X_train.shape)

def load_yolov8_data_from_folders(images_folder, labels_folder):
    images = []
    labels = []

    # Przechodzimy przez wszystkie pliki w folderze obrazów
    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg'):  # Zakładamy, że obrazy mają rozszerzenie .jpg
            image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(labels_folder, filename.split('.')[0] + '.txt')

            # Sprawdzamy, czy istnieje plik tekstowy z etykietami dla obrazu
            if os.path.exists(label_path):
                # Wczytujemy obraz
                image = load_img(image_path, target_size=(
                IMAGE_WIDTH, IMAGE_HEIGHT))  # Zaimportuj funkcję load_img z keras.preprocessing.image
                image_array = img_to_array(image)
                images.append(image_array)

                # Wczytujemy etykiety z pliku tekstowego
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        data = line.split()
                        class_index = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])

                        # Tutaj możesz dodać dodatkowe przetwarzanie danych, jeśli jest to konieczne

                        labels.append([class_index, x_center, y_center, width, height])

    return np.array(images), np.array(labels)

def create_model(input_shape, num_classes):
    # Funkcja tworząca model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_model((IMAGE_WIDTH, IMAGE_HEIGHT, 3), NUM_CLASSES)

# Kompilujemy model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def train_and_save_model(model, X_train, y_train, save_path):
    # Trenowanie modelu
    model.fit(X_train, y_train, epochs=200)  # Możesz dostosować liczbę epok do swoich potrzeb

    # Zapisanie modelu do pliku
    tf.keras.models.save_model(model, save_path)

    print("Model został zapisany do:", save_path)

# Przygotowanie danych treningowych

# Trenowanie i zapisanie modelu
train_and_save_model(model, X_train, y_train, 'trained_model.keras')
