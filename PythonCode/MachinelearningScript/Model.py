import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Parametry modelu
input_shape = (img_height, img_width, 3)
num_classes = 2  # Jeden dla "good_squat", drugi dla "bad_squat"

# Definicja modelu CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Generowanie danych treningowych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,  # Losowy obrót zdjęcia w zakresie od 0 do 90 stopni
)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['good_squat', 'bad_squat'])

# Generowanie danych walidacyjnych
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['good_squat', 'bad_squat'])

# Ustawienie punktu kontrolnego modelu
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Trenowanie modelu
model.fit(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_validation_samples // batch_size,
    callbacks=[checkpoint])
