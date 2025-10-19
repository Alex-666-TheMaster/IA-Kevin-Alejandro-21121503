import cv2 as cv
import numpy as np


img = cv.imread(r'C:\Users\Sears\Documents\Trabajos De IA\figura.png')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)

# --- UMBRALES PARA CADA COLOR ---
# ROJO (tiene dos rangos en HSV)
rojo_bajo1 = np.array([0, 100, 100])
rojo_alto1 = np.array([10, 255, 255])
rojo_bajo2 = np.array([170, 100, 100])
rojo_alto2 = np.array([180, 255, 255])

# VERDE
verde_bajo = np.array([35, 100, 100])
verde_alto = np.array([85, 255, 255])

# AZUL
azul_bajo = np.array([90, 100, 100])
azul_alto = np.array([130, 255, 255])

# AMARILLO
amarillo_bajo = np.array([20, 100, 100])
amarillo_alto = np.array([35, 255, 255])

# --- CREAR MÁSCARAS ---
mask_rojo1 = cv.inRange(img_hsv, rojo_bajo1, rojo_alto1)
mask_rojo2 = cv.inRange(img_hsv, rojo_bajo2, rojo_alto2)
mask_rojo = cv.add(mask_rojo1, mask_rojo2)

mask_verde = cv.inRange(img_hsv, verde_bajo, verde_alto)
mask_azul = cv.inRange(img_hsv, azul_bajo, azul_alto)
mask_amarillo = cv.inRange(img_hsv, amarillo_bajo, amarillo_alto)

# --- APLICAR MÁSCARAS ---
res_rojo = cv.bitwise_and(img_rgb, img_rgb, mask=mask_rojo)
res_verde = cv.bitwise_and(img_rgb, img_rgb, mask=mask_verde)
res_azul = cv.bitwise_and(img_rgb, img_rgb, mask=mask_azul)
res_amarillo = cv.bitwise_and(img_rgb, img_rgb, mask=mask_amarillo)

# Tameño de Ventanas
def resize(img, scale=0.4):
    return cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

# Mostra las Imagenes
cv.imshow('Original', resize(cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)))
cv.imshow('Rojo detectado', resize(cv.cvtColor(res_rojo, cv.COLOR_RGB2BGR)))
cv.imshow('Verde detectado', resize(cv.cvtColor(res_verde, cv.COLOR_RGB2BGR)))
cv.imshow('Azul detectado', resize(cv.cvtColor(res_azul, cv.COLOR_RGB2BGR)))
cv.imshow('Amarillo detectado', resize(cv.cvtColor(res_amarillo, cv.COLOR_RGB2BGR)))

# MÁSCARAS EN BLANCO Y NEGRO  
cv.imshow('Mascara Rojo', resize(mask_rojo))
cv.imshow('Mascara Verde', resize(mask_verde))
cv.imshow('Mascara Azul', resize(mask_azul))
cv.imshow('Mascara Amarillo', resize(mask_amarillo))

cv.waitKey(0)
cv.destroyAllWindows()
