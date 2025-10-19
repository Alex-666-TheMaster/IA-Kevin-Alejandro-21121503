
import cv2 as cv
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

base_path = r"C:\Users\Sears\Documents\Trabajos De IA\Emociones"
face_cascade_front = cv.CascadeClassifier(os.path.join(base_path, 'haarcascade_frontalface_alt.xml'))
face_cascade_side = cv.CascadeClassifier(os.path.join(base_path, 'haarcascade_profileface.xml'))
model_path = os.path.join(base_path, 'modelo_emociones.h5')
# === MENÚ PRINCIPAL ===
print("\n--- SISTEMA DE DETECCIÓN DE EMOCIONES ---")
print("1. Capturar imágenes para entrenamiento")
print("2. Probar emociones con modelo entrenado")
op = input("Selecciona una opción (1/2): ")

# OPCIÓN 1: CREAR DATASET
if op == "1":
    nombre_persona = input("👤 Ingresa el nombre de la persona (ej. Alex, Camila): ")
    data_path = os.path.join(base_path, nombre_persona)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Carpeta creada para {nombre_persona}")
    emociones = ["feliz", "triste", "enojado"]
    for emo in emociones:
        emo_path = os.path.join(data_path, emo)
        if not os.path.exists(emo_path):
            os.makedirs(emo_path)
            print(f" Carpeta creada: {emo_path}")
    print("\nSelecciona la emoción que quieres capturar:")
    print("1. Feliz")
    print("2. Triste")
    print("3. Enojado")
    opcion = input("Opción (1/2/3): ")
    if opcion == "1": emocion = "feliz"
    elif opcion == "2": emocion = "triste"
    elif opcion == "3": emocion = "enojado"
    else:
        print(" Opción inválida.")
        exit()
    output_path = os.path.join(data_path, emocion)
    print(f"\n Capturando rostros de {nombre_persona} con emoción: {emocion.upper()}")
    print("\nSelecciona la fuente de video:")
    print("1. Cámara en vivo")
    print("2. Archivo de video")
    fuente = input("Opción (1/2): ")
    if fuente == "1":
        cap = cv.VideoCapture(0)
    elif fuente == "2":
        ruta_video = input("Ruta del video (ej. C:\\Users\\Sears\\Videos\\video.mp4): ")
        cap = cv.VideoCapture(ruta_video)
    else:
        print(" Opción inválida.")
        exit()
    count = 0
    max_imgs = 5000
    while True:
        ret, frame = cap.read()
        if not ret:
            # Reiniciar video si terminó
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade_front.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            faces = face_cascade_side.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv.resize(rostro, (64, 64), interpolation=cv.INTER_CUBIC)  
            file_name = f"{nombre_persona}_{emocion}_{count}.jpg"
            cv.imwrite(os.path.join(output_path, file_name), rostro)
            count += 1
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, f"{nombre_persona} - {emocion} {count}", (x, y-10), 1, 1.5, (255, 255, 0), 1)
        cv.imshow(f'Capturando {nombre_persona} ({emocion})', frame)
        k = cv.waitKey(1)
       
        if k == ord('q') or count >= max_imgs:
            break
    print(f"\n Captura finalizada para {nombre_persona} ({emocion}). Total imágenes: {count}")
    cap.release()
    cv.destroyAllWindows()

elif op == "2":
    if not os.path.exists(model_path):
        print("No se encontró el modelo 'modelo_emociones.h5'. Entrénalo primero.")
        exit()
    print("\nSelecciona la fuente de video:")
    print("1. Cámara en vivo")
    print("2. Archivo de video")
    fuente = input("Opción (1/2): ")
    if fuente == "1":
        cap = cv.VideoCapture(0)
    elif fuente == "2":
        ruta_video = input("Ruta del video (ej. C:\\Users\\Hellsing\\Videos\\video.mp4): ")
        cap = cv.VideoCapture(ruta_video)
    else:
        print("Opción inválida.")
        exit()
    # Cargar modelo
    modelo = load_model(model_path)
    etiquetas = ['Feliz', 'Triste', 'Enojado']
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade_front.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            faces = face_cascade_side.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv.resize(rostro, (64, 64))  
            rostro = rostro.astype("float") / 255.0
            rostro = img_to_array(rostro)
            rostro = np.expand_dims(rostro, axis=0)
            pred = modelo.predict(rostro, verbose=0)[0]
            emotion_label = etiquetas[np.argmax(pred)]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, emotion_label, (x, y-10), 1, 2, (255, 255, 0), 2)
        cv.imshow('Prueba de emociones', frame)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
else:
    print(" Opción no válida.")
