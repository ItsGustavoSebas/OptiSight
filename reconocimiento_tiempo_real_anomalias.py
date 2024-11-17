import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo sin compilar
autoencoder = tf.keras.models.load_model('autoencoder_model.h5', compile=False)

# Compilar el modelo (opcional si solo usas predict())
autoencoder.compile(optimizer='adam', loss='mse')

def real_time_anomaly_detection():
    cap = cv2.VideoCapture(0)
    threshold = 0.006  # Ajusta este valor según tus necesidades

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img_norm = img / 255.0
        img_norm = np.expand_dims(img_norm, axis=0)

        reconstructed = autoencoder.predict(img_norm)
        error = np.mean(np.square(img_norm - reconstructed))

        print(f"Error de reconstrucción: {error}")

        if error > threshold:
            label = "Anomalía detectada"
            color = (0, 0, 255)  # Rojo
        else:
            label = "Normal"
            color = (0, 255, 0)  # Verde

        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Detección de Anomalías', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_anomaly_detection()
