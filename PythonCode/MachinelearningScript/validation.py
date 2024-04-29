import os
import yaml
import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    pass


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


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


# Wczytanie modelu
model = tf.keras.models.load_model('trained_model.keras')

# Ścieżka do folderu z danymi walidacyjnymi
validation_folder_path = "C:/Users/Ewa/Documents/GitHub/Machine-Learning/PythonCode/MachinelearningScript"


# Wczytanie danych walidacyjnych
X_valid, y_valid = load_yaml_data(validation_folder_path)

# Ocena modelu na danych walidacyjnych
loss, accuracy = model.evaluate(X_valid, y_valid)

print("Strata na danych walidacyjnych:", loss)
print("Dokładność na danych walidacyjnych:", accuracy)
