import os
import cv2
import numpy as np

def load_images(image_folder, img_size=(224, 224)):
    images = []
    labels = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                
                # Determinar la etiqueta basada en la carpeta
                if "no_objeto" in root:
                    label = 0
                elif "objeto" in root:
                    label = 1
                else:
                    continue  # Ignorar imágenes que no están en 'objeto' o 'no_objeto'
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Advertencia: No se pudo cargar la imagen {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalizar a [0, 1]
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)