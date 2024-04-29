import cv2
import numpy as np
import tensorflow as tf

# Wczytanie wytrenowanego modelu
model = tf.keras.models.load_model('trained_model.keras')
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
# Definicja klas
classes = ['1', '2', '3', '4', '5']  # Zastąp nazwami swoimi klasami

# Funkcja do analizowania obrazu z kamery
# Funkcja do analizowania obrazu z kamery
def analyze_camera_image():
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)

    while True:
        # Wczytanie obrazu z kamery
        ret, frame = cap.read()

        # Konwersja obrazu do formatu RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Przygotowanie obrazu do przekazania do modelu
        resized_frame = cv2.resize(rgb_frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        input_image = np.expand_dims(resized_frame, axis=0)

        # Predykcja klas na podstawie obrazu
        predictions = model.predict(input_image)

        # Znalezienie klasy z najwyższym prawdopodobieństwem
        predicted_class_index = np.argmax(predictions)

        # Ograniczenie wartości etykiet do zakresu [0, 4]
        predicted_class_index = np.clip(predicted_class_index, 0, 4)

        predicted_class = classes[predicted_class_index]

        # Wyświetlenie wyniku na obrazie
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Wyświetlenie obrazu z kamery
        cv2.imshow('Camera', frame)

        # Przerwanie pętli po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zamknięcie kamery i zniszczenie okien
    cap.release()
    cv2.destroyAllWindows()


# Wywołanie funkcji do analizy obrazu z kamery
analyze_camera_image()
