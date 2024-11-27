import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time

class ImageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de Imágenes")
        self.root.geometry("800x600")
        
        # Variables de la interfaz
        self.label_var = tk.StringVar(value="objeto")
        self.num_images_var = tk.IntVar(value=50)
        self.save_dir_var = tk.StringVar(value="imagenes")
        self.capture_indices_var = tk.StringVar(value="")  # Para especificar cámaras a capturar
        
        self.all_cameras = {}  # Todas las cámaras detectadas
        self.frames = {}   # Almacenar etiquetas para mostrar los videos
        self.capturing = False  # Bandera para la captura
        self.img_counts = {}  # Contador de imágenes por cámara
        self.update_job = None  # Identificador de la actualización programada
        
        # Título principal
        title = tk.Label(root, text="Captura de Imágenes", font=("Helvetica", 18))
        title.pack(pady=10)
        
        # Configuración de parámetros
        config_frame = tk.Frame(root)
        config_frame.pack(pady=10)
        
        tk.Label(config_frame, text="Índices de cámaras para capturar (separados por comas):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(config_frame, textvariable=self.capture_indices_var).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(config_frame, text="Etiqueta:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(config_frame, textvariable=self.label_var).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(config_frame, text="Cantidad de imágenes por cámara:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(config_frame, textvariable=self.num_images_var).grid(row=2, column=1, padx=5, pady=5)
        
        tk.Label(config_frame, text="Carpeta de guardado:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(config_frame, textvariable=self.save_dir_var).grid(row=3, column=1, padx=5, pady=5)
        
        # Botones de acción
        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Iniciar Captura", command=self.start_capture).grid(row=0, column=0, padx=10)
        tk.Button(button_frame, text="Detener Captura", command=self.stop_capture).grid(row=0, column=1, padx=10)
        tk.Button(button_frame, text="Salir", command=self.on_closing).grid(row=0, column=2, padx=10)
        
        # Contenedor de videos
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(pady=10, expand=True, fill="both")
        
        # Detectar y abrir cámaras disponibles
        self.detect_and_open_cameras()
        
        # Iniciar la actualización de los feeds de video
        self.update_video_feeds()
        
        # Manejar el cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def detect_and_open_cameras(self):
        max_cameras = 10  # Número máximo de cámaras a comprobar
        for idx in range(max_cameras):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.all_cameras[idx] = cap
                
                # Crear un frame para la cámara
                cam_frame = tk.Frame(self.video_frame, bd=2, relief=tk.SOLID)
                cam_frame.pack(side="left", padx=10, pady=10)
                
                # Etiqueta con el índice de la cámara
                label_widget = tk.Label(cam_frame, text=f"Cámara {idx}", font=("Helvetica", 12))
                label_widget.pack()
                
                # Etiqueta para mostrar el video
                video_label = tk.Label(cam_frame)
                video_label.pack()
                self.frames[idx] = video_label
            else:
                cap.release()
    
    def start_capture(self):
        if self.capturing:
            messagebox.showwarning("Advertencia", "La captura ya está en curso.")
            return
        
        label = self.label_var.get().strip()
        if not label:
            messagebox.showerror("Error", "Debes especificar una etiqueta.")
            return
        
        num_images = self.num_images_var.get()
        if num_images <= 0:
            messagebox.showerror("Error", "La cantidad de imágenes debe ser mayor a 0.")
            return
        
        save_dir = self.save_dir_var.get().strip()
        if not save_dir:
            messagebox.showerror("Error", "Debes especificar una carpeta para guardar las imágenes.")
            return
        
        # Obtener los índices de las cámaras a capturar
        capture_indices = self.capture_indices_var.get()
        if not capture_indices:
            messagebox.showerror("Error", "Debes especificar al menos una cámara para capturar.")
            return
        
        try:
            capture_indices = list(map(int, capture_indices.split(",")))
        except ValueError:
            messagebox.showerror("Error", "Los índices de las cámaras deben ser números separados por comas.")
            return
        
        # Verificar que las cámaras especificadas estén disponibles
        self.cameras_to_capture = {}
        for idx in capture_indices:
            if idx in self.all_cameras:
                self.cameras_to_capture[idx] = self.all_cameras[idx]
                self.img_counts[idx] = 0  # Inicializar contador de imágenes para esta cámara
            else:
                messagebox.showerror("Error", f"La cámara {idx} no está disponible.")
                return
        
        if not self.cameras_to_capture:
            messagebox.showerror("Error", "No se encontraron las cámaras especificadas para capturar.")
            return
        
        # Crear carpetas para guardar imágenes
        for idx in self.cameras_to_capture.keys():
            cam_folder = os.path.join(save_dir, f"camara_{idx}", label)
            os.makedirs(cam_folder, exist_ok=True)
        
        self.capturing = True
        
        # Iniciar el bucle de captura en un hilo separado
        self.capture_thread = threading.Thread(target=self.capture_loop, args=(label, num_images, save_dir))
        self.capture_thread.start()
    
    def update_video_feeds(self):
        if not self.all_cameras:
            return
        for idx, cap in self.all_cameras.items():
            ret, frame = cap.read()
            if ret:
                # Si está capturando y esta cámara está en las seleccionadas, agregar "Capturando" al frame
                if self.capturing and idx in self.cameras_to_capture:
                    cv2.putText(frame, "Capturando", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Redimensionar y convertir el frame para Tkinter
                frame_resized = cv2.resize(frame, (320, 240))
                cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.frames[idx].imgtk = imgtk
                self.frames[idx].configure(image=imgtk)
            else:
                # Si no se pudo leer el frame, eliminar la cámara
                cap.release()
                self.all_cameras.pop(idx)
                self.frames[idx].configure(text=f"Cámara {idx} - Error")
                self.frames.pop(idx)
        
        # Programar la siguiente actualización y almacenar el identificador
        self.update_job = self.root.after(30, self.update_video_feeds)
    
    def capture_loop(self, label, num_images, save_dir):
        while self.capturing:
            all_cameras_done = True
            for idx, cap in self.cameras_to_capture.items():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Guardar las imágenes
                if self.img_counts.get(idx, 0) < num_images:
                    cam_folder = os.path.join(save_dir, f"camara_{idx}", label)
                    img_name = os.path.join(cam_folder, f"{label}_{self.img_counts[idx]}.jpg")
                    cv2.imwrite(img_name, frame)
                    self.img_counts[idx] += 1
                
                # Verificar si esta cámara ha terminado
                if self.img_counts[idx] < num_images:
                    all_cameras_done = False
            if all_cameras_done:
                self.capturing = False
                messagebox.showinfo("Finalizado", "Captura de imágenes completada.")
                break
            else:
                time.sleep(0.01)  # Pequeña pausa para evitar alto uso de CPU
    
    def stop_capture(self):
        if self.capturing:
            self.capturing = False
            messagebox.showinfo("Información", "Captura detenida.")
        # No cerramos las cámaras ni detenemos el feed de video, solo detenemos la captura
    
    def on_closing(self):
        # Detener la captura y liberar recursos
        self.capturing = False
        # Cancelar la actualización programada de video feeds
        if hasattr(self, 'update_job') and self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        # Liberar cámaras
        for cap in self.all_cameras.values():
            cap.release()
        self.all_cameras.clear()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()
