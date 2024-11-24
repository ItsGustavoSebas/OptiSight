import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image, ImageTk
import os
import threading
from queue import Queue, Empty  # Importar la excepción Empty
import queue

# Cargar los modelos
modelo_objeto = tf.keras.models.load_model('modelo_objeto.h5')
autoencoder = tf.keras.models.load_model('autoencoder_model.h5', compile=False)
autoencoder.compile(optimizer='adam', loss='mse')

class CameraThread(threading.Thread):
    def __init__(self, idx, cap, models, result_queue, stop_event):
        threading.Thread.__init__(self)
        self.idx = idx
        self.cap = cap
        self.models = models  # Diccionario con los modelos
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Cámara {self.idx} dejó de responder. Eliminándola de la lista activa.")
                self.cap.release()
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.background_subtractor.apply(frame_gray)

            # Umbralizar la máscara para eliminar sombras y ruido
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

            # Encontrar contornos en la máscara
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Obtener cuadros delimitadores de objetos en movimiento
            object_bounding_boxes = []
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Umbral para eliminar ruido
                    x, y, w, h = cv2.boundingRect(contour)
                    object_bounding_boxes.append((x, y, w, h))

            # Procesamiento del frame
            img = cv2.resize(frame, (224, 224))
            img_norm = img / 255.0
            img_norm_expanded = np.expand_dims(img_norm, axis=0)

            # Predicción del autoencoder
            reconstructed = self.models['autoencoder'].predict(img_norm_expanded)
            error = np.mean(np.square(img_norm_expanded - reconstructed))

            difference = np.abs(img_norm - reconstructed[0])
            difference_gray = cv2.cvtColor((difference * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(difference_gray, 50, 255, cv2.THRESH_BINARY)
            anomaly_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            anomaly_bounding_boxes = []
            for contour in anomaly_contours:
                if cv2.contourArea(contour) > 100:  # Umbral para eliminar ruido
                    x, y, w, h = cv2.boundingRect(contour)
                    anomaly_bounding_boxes.append((x, y, w, h))

            # Escalar los cuadros a las dimensiones originales
            scale_x = frame.shape[1] / 224
            scale_y = frame.shape[0] / 224
            scaled_anomaly_boxes = []
            for x, y, w, h in anomaly_bounding_boxes:
                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)
                w_scaled = int(w * scale_x)
                h_scaled = int(h * scale_y)
                scaled_anomaly_boxes.append((x_scaled, y_scaled, w_scaled, h_scaled))

            # Predicción del modelo de objeto
            obj_prediction = self.models['modelo_objeto'].predict(img_norm_expanded)[0][0]

            # Poner los resultados en la cola
            self.result_queue.put({
                'idx': self.idx,
                'frame': frame,
                'error': error,
                'obj_prediction': obj_prediction,
                'anomaly_bounding_boxes': scaled_anomaly_boxes,
                'object_bounding_boxes': object_bounding_boxes,
            })
def calculate_anomaly_threshold(autoencoder, X_train, percentile=95):
    errors = []
    for img in X_train:
        img = np.expand_dims(img, axis=0)
        reconstructed = autoencoder.predict(img)
        error = np.mean(np.square(img - reconstructed))
        errors.append(error)
    threshold = np.percentile(errors, percentile)
    print(f"Umbral de anomalía calculado automáticamente (percentil {percentile}): {threshold}")
    return threshold

def check_cameras(camera_indices):
    available_cameras = []
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available_cameras.append(idx)
        else:
            print(f"Cámara {idx} no disponible.")
        cap.release()
    return available_cameras

def load_images(folder):
    """
    Carga imágenes desde una carpeta con la estructura especificada y devuelve las imágenes y sus etiquetas.
    - Cada carpeta principal (e.g., `camara 0`, `camara N`) corresponde a una cámara.
    - Dentro de estas, las subcarpetas (`no_objeto`, `objeto`) representan las clases.
    
    Args:
        folder (str): Ruta de la carpeta principal.

    Returns:
        tuple: 
            - X (numpy array): Imágenes procesadas (normalizadas y redimensionadas).
            - y (numpy array): Etiquetas correspondientes (0 para `no_objeto`, 1 para `objeto`).
    """
    X = []
    y = []
    
    for camera_folder in os.listdir(folder):
        camera_path = os.path.join(folder, camera_folder)
        if not os.path.isdir(camera_path):
            continue
        
        for class_folder in os.listdir(camera_path):
            class_path = os.path.join(camera_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # Etiqueta basada en la subcarpeta: `no_objeto` -> 0, `objeto` -> 1
            if class_folder.lower() == 'no_objeto':
                label = 0
            elif class_folder.lower() == 'objeto':
                label = 1
            else:
                continue  # Ignorar carpetas desconocidas
            
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                if not image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                    continue
                
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Redimensionar y normalizar la imagen
                image = cv2.resize(image, (224, 224))
                image = image / 255.0  # Normalizar entre 0 y 1
                X.append(image)
                y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Objetos y Anomalías")
        self.anomaly_multipliers = {}  # Almacena los multiplicadores por cámara
        self.anomaly_percentages = {}  # Almacena los porcentajes por cámara
        
        # Etiqueta principal
        label = tk.Label(root, text="Detección de Objetos y Anomalías", font=("Helvetica", 16))
        label.pack(pady=10)
        
        self.camera_threads = {}  # Diccionario para almacenar los hilos de las cámaras
        self.result_queue = Queue()  # Cola para recibir los resultados de los hilos
        self.stop_event = threading.Event()  # Evento para detener los hilos

        # Controles de sensibilidad
        sensitivity_frame = tk.Frame(root)
        sensitivity_frame.pack(pady=10)
        
        tk.Label(sensitivity_frame, text="Sensibilidad de detección de objeto:").grid(row=0, column=0, sticky="e")
        self.object_sensitivity = tk.DoubleVar(value=0.5)
        tk.Scale(sensitivity_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.object_sensitivity).grid(row=0, column=1)
        
        tk.Label(sensitivity_frame, text="Multiplicador de umbral de anomalía:").grid(row=1, column=0, sticky="e")
        self.anomaly_multiplier = tk.DoubleVar(value=1.0)
        tk.Scale(sensitivity_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.anomaly_multiplier).grid(row=1, column=1)

        tk.Label(sensitivity_frame, text="Porcentaje del multiplicador de umbral de anomalía:").grid(row=2, column=0, sticky="e")
        self.anomaly_percentage = tk.DoubleVar(value=100.0)
        tk.Scale(sensitivity_frame, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, variable=self.anomaly_percentage).grid(row=2, column=1)
        
        self.update_anomaly_button = tk.Button(sensitivity_frame, text="Actualizar Umbral de Anomalía", command=self.update_anomaly_threshold)
        self.update_anomaly_button.grid(row=1, column=2, padx=5)
        
        # Cantidad de cámaras
        tk.Label(root, text="Cantidad de cámaras a usar:").pack()
        self.camera_count = tk.IntVar(value=1)
        tk.Entry(root, textvariable=self.camera_count).pack()
        
        # Botones de iniciar y reiniciar
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="Iniciar", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.restart_button = tk.Button(button_frame, text="Reiniciar Cámaras", command=self.restart_cameras)
        self.restart_button.grid(row=0, column=1, padx=5)
        
        # Contenedor de videos
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(pady=10)
        
        # Inicializar variables
        self.caps = {}
        self.running = False
        self.anomaly_threshold = None
        self.X_train = None
        self.video_labels = {}  # Diccionario para almacenar etiquetas de cámaras
        self.info_labels = {}  # Diccionario para etiquetas de información
        self.camera_frames = {}
        
    def start_detection(self):
        if self.running:
            messagebox.showwarning("Advertencia", "La detección ya está en ejecución.")
            return
        
        self.object_threshold = self.object_sensitivity.get()
        anomaly_percentile = 95  # Puedes ajustar el percentil aquí
        self.num_cameras = self.camera_count.get()
        
        # Cargar imágenes y calcular el umbral de anomalía
        image_folder = './imagenes'
        X_train, y_train = load_images(image_folder)
        if X_train is None or y_train is None:
            messagebox.showerror("Error", "Error al cargar las imágenes de entrenamiento.")
            return
        self.X_train = X_train[y_train == 1]  # Usar imágenes normales para calcular el umbral
        self.base_anomaly_threshold = calculate_anomaly_threshold(autoencoder, self.X_train, percentile=anomaly_percentile)
        
        # Inicializar cámaras
        camera_indices = list(range(self.num_cameras))
        self.camera_indices = check_cameras(camera_indices)
        if not self.camera_indices:
            messagebox.showerror("Error", "No se encontraron cámaras disponibles.")
            return
        
        self.running = True
        self.caps = {idx: cv2.VideoCapture(idx) for idx in self.camera_indices}
        
        # Crear hilos para cada cámara
        self.camera_threads = {}
        for idx in self.camera_indices:
            cap = self.caps[idx]
            camera_thread = CameraThread(
                idx=idx,
                cap=cap,
                models={'autoencoder': autoencoder, 'modelo_objeto': modelo_objeto},
                result_queue=self.result_queue,
                stop_event=self.stop_event
            )
            camera_thread.start()
            self.camera_threads[idx] = camera_thread

            camera_frame = tk.Frame(self.video_frame, bd=2, relief=tk.SOLID, padx=5, pady=5)
            camera_frame.grid(row=idx // 2, column=idx % 2, padx=5, pady=5)
            self.camera_frames[idx] = camera_frame
            controls_frame = tk.Frame(camera_frame)
            controls_frame.pack(pady=5)
            self.anomaly_multipliers[idx] = tk.DoubleVar(value=1.0)
            self.anomaly_percentages[idx] = tk.DoubleVar(value=100.0)

            tk.Label(controls_frame, text="Multiplicador de umbral:").grid(row=0, column=0, sticky="e")
            tk.Scale(
                controls_frame,
                from_=0.1,
                to=3.0,
                resolution=0.1,
                orient=tk.HORIZONTAL,
                variable=self.anomaly_multipliers[idx]
            ).grid(row=0, column=1)
    
            # Porcentaje del multiplicador de umbral de anomalía
            tk.Label(controls_frame, text="Porcentaje del multiplicador:").grid(row=1, column=0, sticky="e")
            tk.Scale(
                controls_frame,
                from_=0,
                to=100,
                resolution=1,
                orient=tk.HORIZONTAL,
                variable=self.anomaly_percentages[idx]
            ).grid(row=1, column=1)

            # Crear la etiqueta de información dentro del Frame de la cámara
            info_label = tk.Label(
                camera_frame,
                text=f"Cámara {idx} - Error: --- , Umbral: ---",
                font=("Helvetica", 10),
                anchor="w",
                justify="left",
                wraplength=300,
            )
            info_label.pack()  # Empaquetar la etiqueta en el frame
            self.info_labels[idx] = info_label

            # Crear la etiqueta del video dentro del Frame de la cámara
            label = tk.Label(camera_frame)
            label.pack()  # Empaquetar el video en el frame
            self.video_labels[idx] = label

        # Iniciar el bucle de actualización de la interfaz
        self.update_gui()
        
        
    def restart_cameras(self):
        if not self.running:
            messagebox.showwarning("Advertencia", "La detección no está en ejecución.")
            return
        self.running = False
        self.stop_event.set()
        for thread in self.camera_threads.values():
            thread.join()
        self.camera_threads = {}
        for cap in self.caps.values():
            cap.release()
        self.caps = {}
        self.stop_event.clear()

        # Eliminar todos los widgets de cámaras anteriores
        for idx in list(self.video_labels.keys()):
            self.video_labels[idx].destroy()
        self.video_labels.clear()
        
        for idx in list(self.info_labels.keys()):
            self.info_labels[idx].destroy()
        self.info_labels.clear()
        
        for idx in list(self.camera_frames.keys()):
            self.camera_frames[idx].destroy()
        self.camera_frames.clear()
                
        self.start_detection()

        
    def update_anomaly_threshold(self):
        if self.X_train is None:
            messagebox.showerror("Error", "No se han cargado los datos de entrenamiento.")
            return
        anomaly_percentile = 95  # Puedes ajustar el percentil aquí
        self.base_anomaly_threshold = calculate_anomaly_threshold(autoencoder, self.X_train, percentile=anomaly_percentile)
        messagebox.showinfo("Información", f"Umbral de anomalía recalculado: {self.base_anomaly_threshold}")
        
    def update_gui(self):
        try:
            while True:
                result = self.result_queue.get_nowait()
                idx = result['idx']
                frame = result['frame']
                error = result['error']
                obj_prediction = result['obj_prediction']
                # anomaly_bounding_boxes = result.get('anomaly_bounding_boxes', [])
                object_bounding_boxes = result.get('object_bounding_boxes', [])
                # Ajuste del umbral de anomalía mediante multiplicador
                anomaly_multiplier = self.anomaly_multipliers[idx].get()
                anomaly_percentage = self.anomaly_percentages[idx].get() / 100.0
                anomaly_threshold = self.base_anomaly_threshold * anomaly_multiplier * anomaly_percentage
    
                error = round(error, 6)
                anomaly_threshold = round(anomaly_threshold, 6)
    
                anomaly_label = "Anomalía detectada" if error > anomaly_threshold else "Normal"
                anomaly_color = (0, 0, 255) if error > anomaly_threshold else (0, 255, 0)
    
                self.info_labels[idx].config(text=f"Cámara {idx} - Error: {error:.6f}, Umbral: {anomaly_threshold:.6f}")
    
                anomaly_detected = error > anomaly_threshold   

                object_threshold = self.object_sensitivity.get()
                if obj_prediction > object_threshold:
                    obj_label = "Objeto detectado"
                    obj_color = (0, 255, 0)  # Verde
                else:
                    obj_label = "Objeto no detectado"
                    obj_color = (0, 0, 255)  # Rojo
    
                # Agregar texto al frame
                cv2.putText(frame, obj_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, obj_color, 2)
                cv2.putText(frame, anomaly_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, anomaly_color, 2)
    
                # Mostrar el error y el umbral en la consola
                print(f"Cámara {idx} - Error: {error:.6f}, Umbral: {anomaly_threshold:.6f}")

                if anomaly_detected:
                    # Dibujar cuadros de objetos detectados
                    for x, y, w, h in object_bounding_boxes:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
                    # # Dibujar cuadros de anomalías detectadas
                    # for x, y, w, h in anomaly_bounding_boxes:
                    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


                # Redimensionar el frame para mostrarlo en la interfaz
                frame_resized = cv2.resize(frame, (320, 240))  # Ajusta el tamaño según tus necesidades
    
                # Convertir el frame a formato compatible con Tkinter
                cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_labels[idx].imgtk = imgtk
                self.video_labels[idx].configure(image=imgtk)

                self.info_labels[idx].config(
                    text=f"Cámara {idx} - Error: {error:.6f}, Umbral: {anomaly_threshold:.6f}"
                )

        except queue.Empty:
            pass

        if self.running:
            self.root.after(10, self.update_gui)
        else:
            self.stop_event.set()
            for idx, thread in self.camera_threads.items():
                thread.join()
            for cap in self.caps.values():
                cap.release()
            self.caps = {}

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
