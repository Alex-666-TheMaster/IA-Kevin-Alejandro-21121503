import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Tamaño inicial del cuadrado
rect_size = 150
center_x, center_y = 320, 240

# Ángulo inicial y suavizado
angle_smoothed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            # Coordenadas del índice (8) y del pulgar (4)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Dibuja la línea entre el índice y el pulgar
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)

            # Calcular distancia entre dedos
            distance = math.hypot(x2 - x1, y2 - y1)
            rect_size = int(max(50, min(300, distance * 4)))

            # Calcular ángulo entre los dedos (en grados)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Suavizar el ángulo para rotación estable
            angle_smoothed = angle_smoothed * 0.85 + angle * 0.15

            # Dibujar los landmarks de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calcular vértices del cuadrado girado
    half = rect_size // 2
    points = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]
    ])

    # Matriz de rotación
    rot_mat = cv2.getRotationMatrix2D((0, 0), angle_smoothed, 1)
    rotated_points = np.dot(points, rot_mat[:, :2].T)

    # Trasladar al centro de la pantalla
    rotated_points += np.array([center_x, center_y])

    # Convertir a enteros para dibujar
    rotated_points = rotated_points.astype(int)

    # Dibujar el cuadrado rotado
    cv2.polylines(frame, [rotated_points], isClosed=True, color=(255, 0, 255), thickness=3)

    cv2.imshow("Zoom y rotacion con la mano", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()





