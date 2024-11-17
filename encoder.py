import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Función para cargar imágenes etiquetadas
def load_images(image_folder, img_size=(224, 224)):
    images = []
    labels = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)

                # Determinar la etiqueta
                if "no_objeto" in root:
                    label = 0  # Sin objeto
                elif "objeto" in root:
                    label = 1  # Con objeto
                else:
                    continue  # Ignorar imágenes sin etiqueta válida

                # Cargar y procesar la imagen
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Advertencia: No se pudo cargar la imagen {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalizar a [0, 1]
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Definir el modelo de autoencoder
def create_autoencoder():
    input_img = Input(shape=(224, 224, 3))

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Entrenar el autoencoder
def train_autoencoder(image_folder, epochs=30, batch_size=32):
    X, y = load_images(image_folder)

    # Filtrar imágenes etiquetadas como "objeto" (con el objeto presente)
    X_objeto = X[y == 1]

    # Mezclar datos
    np.random.shuffle(X_objeto)

    # Dividir en entrenamiento y validación (80% entrenamiento, 20% validación)
    split_idx = int(0.8 * len(X_objeto))
    X_train = X_objeto[:split_idx]
    X_val = X_objeto[split_idx:]

    # Crear el modelo de autoencoder
    autoencoder = create_autoencoder()

    # Entrenar el modelo
    autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val)
    )

    # Guardar el modelo
    autoencoder.save('autoencoder_model.h5')
    print("Modelo de autoencoder guardado como 'autoencoder_model.h5'.")

    return autoencoder

# Evaluar el modelo y mostrar reconstrucciones
def evaluate_autoencoder(autoencoder, image_folder):
    X, y = load_images(image_folder)

    # Dividir imágenes para evaluación
    X_objeto = X[y == 1]  # Con objeto
    X_no_objeto = X[y == 0]  # Sin objeto

    # Mostrar reconstrucciones de "objeto"
    decoded_objeto = autoencoder.predict(X_objeto)
    n = min(len(X_objeto), 10)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_objeto[i])
        plt.title(f"Original (Obj {i})")
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_objeto[i])
        plt.title(f"Reconstruida (Obj {i})")
        plt.axis("off")
    plt.show()

    # Calcular errores de reconstrucción
    print("\nErrores de reconstrucción:")
    for i, img in enumerate(X_no_objeto[:10]):  # Mostrar hasta 10 imágenes de "no_objeto"
        reconstructed = autoencoder.predict(np.expand_dims(img, axis=0))
        error = np.mean(np.square(img - reconstructed[0]))
        print(f"Imagen {i} (no_objeto): Error de reconstrucción = {error:.4f}")

if __name__ == "__main__":
    image_folder = './imagenes'

    # Entrenar el autoencoder
    autoencoder = train_autoencoder(image_folder, epochs=20, batch_size=32)

    # Evaluar el modelo
    evaluate_autoencoder(autoencoder, image_folder)