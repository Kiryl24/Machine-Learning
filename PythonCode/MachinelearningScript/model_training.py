import os
import yaml
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def load_yaml_data(folder_path):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.yaml'):  # Zakładamy, że dane są w plikach z rozszerzeniem .yaml
            yaml_path = os.path.join(folder_path, filename)

            # Wczytaj dane YAML
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)

                # Sprawdź, czy dane zawierają klucz "image" i "label"
                if 'image' in data and 'label' in data:
                    # Wczytaj obraz z pliku lub z adresu URL
                    # Tutaj dodaj kod odpowiedzialny za wczytanie obrazu
                    # Zakładamy, że obrazy są podane jako ścieżki do plików
                    image_path = data['image']
                    image = load_img(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
                    image_array = img_to_array(image)
                    images.append(image_array)

                    # Wczytaj etykietę
                    label = data['label']
                    labels.append(label)

    return np.array(images), np.array(labels)


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
