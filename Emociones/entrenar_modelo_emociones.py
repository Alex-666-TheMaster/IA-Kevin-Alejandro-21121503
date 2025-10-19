import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

base_path = r"C:\Users\Sears\Documents\Trabajos De IA\Emociones"
img_size = (64, 64)
batch_size = 32
epochs = 50
# === GENERADOR DE DATOS ===
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)
train_generator = datagen.flow_from_directory(
    base_path,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
val_generator = datagen.flow_from_directory(
    base_path,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
# === MODELO CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])
# === COMPILAR ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# === ENTRENAR ===
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
    
)
model.save(os.path.join(base_path, "modelo_emociones.h5"))
print(" Modelo entrenado y guardado correctamente en 'modelo_emociones.h5'")
