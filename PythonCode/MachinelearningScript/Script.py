import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

# Wczytanie wytrenowanego modelu
model = load_model("trained_model.keras")  # Upewnij się, że plik trained_model.keras znajduje się w tym samym katalogu co ten skrypt

# Definicja funkcji do przetwarzania klatki obrazu
def process_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Wymiary obrazu zgodne z tymi używanymi podczas treningu
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)  # Ustawienie wartości 0 oznacza wybór pierwszej dostępnej kamery

# Pętla analizująca obrazy z kamery
while True:
    ret, frame = cap.read()  # Odczytanie klatki z kamery
    if not ret:
        print("Nie można odczytać klatki z kamery!")
        break

    # Przetwarzanie klatki obrazu
    processed_frame = process_frame(frame)

    # Przewidywanie za pomocą wytrenowanego modelu
    prediction = model.predict(processed_frame)

    # Wyświetlenie wyniku
    if prediction > 0.5:  # Próg 0.5 dla klasyfikacji binarnej
        cv2.putText(frame, "Przysiad poprawny", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Przysiad niepoprawny", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Wyświetlenie klatki z kamery
    cv2.imshow("Analiza przysiadu", frame)

    # Sprawdzenie czy użytkownik nacisnął klawisz 'q' (quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zamknięcie kamery i zniszczenie okien
cap.release()
cv2.destroyAllWindows()
