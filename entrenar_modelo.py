import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle 
import cv2

# Función para cargar imágenes con etiquetas correctas
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
                    continue  # Ignorar si no está en 'objeto' o 'no_objeto'

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Advertencia: No se pudo cargar la imagen {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalizar a [0, 1]
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Crear directorios para guardar imágenes
def create_dirs(base_dir):
    correct_dir = os.path.join(base_dir, "correctas")
    incorrect_dir = os.path.join(base_dir, "incorrectas")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    return correct_dir, incorrect_dir

# Guardar imágenes en carpetas correspondientes
def save_images(images, predictions, true_labels, correct_dir, incorrect_dir):
    for i, (image, pred, true_label) in enumerate(zip(images, predictions, true_labels)):
        # Determinar la etiqueta predicha
        label_pred = 1 if pred > 0.5 else 0

        # Depuración: imprimir información detallada
        print(f"Imagen {i}: predicción={label_pred}, probabilidad={pred:.2f}, etiqueta verdadera={true_label}")

        # Lógica personalizada: todas las imágenes de clase 0 van a incorrectas
        if true_label == 0:
            folder = incorrect_dir
            print(f"Guardando clase 0 en incorrectas: {folder}")
        else:
            folder = correct_dir
            print(f"Guardando en correctas: {folder}")
        # Guardar la imagen
        filename = os.path.join(folder, f"img_{i}_pred_{label_pred}_true_{true_label}_prob_{pred:.2f}.jpg")
        img_to_save = (image * 255).astype(np.uint8)
        cv2.imwrite(filename, img_to_save)





def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
    
if __name__ == "__main__":
    image_folder = './imagenes'
    save_base_dir = './resultados_imagenes'

    
    X_train, y_train = load_images(image_folder)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)  
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  
    )

    train_generator = datagen.flow(X_train, y_train, subset='training', shuffle=True)
    validation_generator = datagen.flow(X_train, y_train, subset='validation', shuffle=True)

    model = create_model()

    model.fit(train_generator, epochs=15, validation_data=validation_generator)

    model.save("modelo_objeto.h5")
    # Crear carpetas para guardar las imágenes clasificadas
    correct_dir, incorrect_dir = create_dirs(save_base_dir)

    # Evaluar el modelo en el conjunto de validación
    for batch in validation_generator:
        images, true_labels = batch
        predictions = model.predict(images).flatten()
        save_images(images, predictions, true_labels, correct_dir, incorrect_dir)
        # Detener el bucle después de un lote completo (opcional para evitar guardar demasiadas imágenes)
        break
    print("Modelo guardado como 'modelo_objeto.h5'")
