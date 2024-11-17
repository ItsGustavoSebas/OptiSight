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
        self.camera_indices_var = tk.StringVar(value="0,1")  # Por defecto, cámaras 0 y 1
        self.save_dir_var = tk.StringVar(value="imagenes")
        
        self.cameras = {}  # Almacenar cámaras activas
        self.frames = {}   # Almacenar etiquetas para mostrar los videos
        self.capturing = False  # Bandera para la captura
        self.img_counts = {}  # Contador de imágenes por cámara
        
        # Título principal
        title = tk.Label(root, text="Captura de Imágenes", font=("Helvetica", 18))
        title.pack(pady=10)
        
        # Configuración de parámetros
        config_frame = tk.Frame(root)
        config_frame.pack(pady=10)
        
        tk.Label(config_frame, text="Índices de cámaras (separados por comas):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(config_frame, textvariable=self.camera_indices_var).grid(row=0, column=1, padx=5, pady=5)
        
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
        
        # Manejar el cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def start_capture(self):
        if self.capturing:
            messagebox.showwarning("Advertencia", "La captura ya está en curso.")
            return
        
        # Limpiar cámaras y widgets anteriores
        self.stop_capture(clear_widgets=True)
        
        camera_indices = self.camera_indices_var.get()
        if not camera_indices:
            messagebox.showerror("Error", "Debes especificar al menos una cámara.")
            return
        
        try:
            camera_indices = list(map(int, camera_indices.split(",")))
        except ValueError:
            messagebox.showerror("Error", "Los índices de las cámaras deben ser números separados por comas.")
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
        
        # Crear carpetas para guardar imágenes y abrir cámaras
        self.frames = {}
        self.cameras = {}
        self.img_counts = {}
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                messagebox.showerror("Error", f"No se pudo abrir la cámara {idx}.")
                continue
            self.cameras[idx] = cap
            self.img_counts[idx] = 0  # Inicializar contador de imágenes para esta cámara
            
            # Crear carpeta para guardar imágenes
            cam_folder = os.path.join(save_dir, f"camara_{idx}", label)
            os.makedirs(cam_folder, exist_ok=True)
            
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
        
        if not self.cameras:
            messagebox.showerror("Error", "No se pudieron abrir las cámaras especificadas.")
            return
        
        self.capturing = True
        
        # Iniciar la actualización de los feeds de video
        self.update_video_feeds()
        
        # Iniciar el bucle de captura en un hilo separado
        self.capture_thread = threading.Thread(target=self.capture_loop, args=(label, num_images, save_dir))
        self.capture_thread.start()
    
    def update_video_feeds(self):
        if not self.cameras:
            return
        for idx, cap in self.cameras.items():
            ret, frame = cap.read()
            if ret:
                # Si está capturando, agregar "Capturando" al frame
                if self.capturing:
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
                self.cameras.pop(idx)
                self.frames[idx].configure(text=f"Cámara {idx} - Error")
                self.frames.pop(idx)
        
        # Programar la siguiente actualización
        self.root.after(30, self.update_video_feeds)
    
    def capture_loop(self, label, num_images, save_dir):
        while self.capturing:
            all_cameras_done = True
            for idx, cap in self.cameras.items():
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
    
    def stop_capture(self, clear_widgets=False):
        if self.capturing:
            self.capturing = False
            messagebox.showinfo("Información", "Captura detenida.")
        # Liberar cámaras
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()
        # Eliminar widgets si es necesario
        if clear_widgets:
            for frame in self.frames.values():
                frame.master.destroy()
            self.frames.clear()
    
    def on_closing(self):
        # Liberar todas las cámaras
        self.stop_capture(clear_widgets=True)
        self.root.destroy()
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()
