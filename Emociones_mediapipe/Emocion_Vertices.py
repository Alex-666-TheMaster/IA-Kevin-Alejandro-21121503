import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Filtro de suavizado
smooth_vals = {"mouth": 0, "eye": 0, "brow": 0}
alpha = 0.6  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    emotion = "Neutral "

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            def p(idx): return np.array([lm[idx].x * w, lm[idx].y * h])

            # Puntos clave
            mouth_top, mouth_bottom = p(13), p(14)
            mouth_left, mouth_right = p(61), p(291)
            eye_left_top, eye_left_bottom = p(159), p(145)
            eye_right_top, eye_right_bottom = p(386), p(374)
            brow_left, brow_left_top = p(336), p(105)
            brow_right, brow_right_top = p(70), p(55)

            face_width = dist(p(234), p(454))

            # Cálculos base
            mouth_open = dist(mouth_top, mouth_bottom) / face_width
            eye_open = (dist(eye_left_top, eye_left_bottom) + dist(eye_right_top, eye_right_bottom)) / (2 * face_width)
            brow_raise = (dist(brow_left, brow_left_top) + dist(brow_right, brow_right_top)) / (2 * face_width)

            # Suavizado
            smooth_vals["mouth"] = alpha * smooth_vals["mouth"] + (1 - alpha) * mouth_open
            smooth_vals["eye"] = alpha * smooth_vals["eye"] + (1 - alpha) * eye_open
            smooth_vals["brow"] = alpha * smooth_vals["brow"] + (1 - alpha) * brow_raise

            m, e, b = smooth_vals["mouth"], smooth_vals["eye"], smooth_vals["brow"]

            # Mostrar valores para debug
            cv2.putText(frame, f"Mouth: {m:.2f}  Eye: {e:.2f}  Brow: {b:.2f}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            #Detección refinada 
            if 0.12 <= m <= 0.16 and 0.05 <= e <= 0.07 and 0.38 <= b <= 0.42:
                emotion = "Enojo "
            elif 0.01 <= m <= 0.03 and 0.03 <= e <= 0.05 and 0.36 <= b <= 0.40:
                emotion = "Tristeza "
            elif 0.09 <= m < 0.16 and abs(b - 0.36) < 0.03:
                emotion = "Felicidad "
            else:
                emotion = "Neutral "

            # Mostrar emoción en pantalla
            cv2.putText(frame, f'Emocion: {emotion}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Dibujar puntos 
            for idx in [13, 14, 61, 291, 159, 145, 386, 374, 336, 105, 70, 55]:
                x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Detección de Emociones (ajustada FINAL)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
