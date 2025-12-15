import os

# Ruta principal del dataset (📁 cambia esto por la ruta donde está tu dataset)
dataset_path = r"C:\Users\Sears\Documents\Trabajos De IA\Proyecto-2\dataset"

# Extensiones de imagen que quieres contar
extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Recorremos las carpetas dentro del dataset
print("Conteo de imágenes por clase:\n")  

for carpeta in os.listdir(dataset_path):
    carpeta_path = os.path.join(dataset_path, carpeta)

    # Verifica que realmente sea una carpeta (no un archivo suelto)
    if os.path.isdir(carpeta_path):
        # Cuenta solo los archivos con extensión de imagen
        contador = 0
        for archivo in os.listdir(carpeta_path):
            _, extension = os.path.splitext(archivo)
            if extension.lower() in extensiones:
                contador += 1

        print(f"📁 {carpeta}: {contador} imágenes")
