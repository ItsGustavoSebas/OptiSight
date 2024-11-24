import tkinter as tk
import subprocess  # Módulo para ejecutar archivos del sistema
from tkinter import messagebox, Scale, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import threading
import os
import shutil
import csv
import pandas as pd 
import matplotlib.pyplot as plt
# Función para verificar las credenciales
def verificar_credenciales():
    usuario = entry_usuario.get()
    contraseña = entry_contraseña.get()
    
    if usuario == "admin" and contraseña == "12345":
        messagebox.showinfo("Login", "¡Inicio de sesión exitoso!")
        ventana.destroy()  # Cierra la ventana del login
        abrir_dashboard()  # Llama a la función para abrir el dashboard
    else:
        messagebox.showerror("Login", "Usuario o contraseña incorrectos.")

# Función para abrir el dashboard
def abrir_dashboard():
    dashboard = tk.Tk()  # Nueva ventana para el dashboard
    dashboard.title("Dashboard")
    dashboard.geometry("1000x800")

     # Imagen de fondo
    # Imagen de fondo
    background_image_path = "background_images/backgroundimage.jpg"
    try:
        # Cargar y redimensionar la imagen
        background_image = Image.open(background_image_path)
        background_image = background_image.resize((1000, 800))
        bg_image = ImageTk.PhotoImage(background_image)

        # Crear un Label para la imagen de fondo
        bg_label = tk.Label(dashboard, image=bg_image)
        bg_label.image = bg_image  # Guardar referencia de la imagen
        bg_label.place(relwidth=1, relheight=1)  # Ajusta al 100% del tamaño

    except FileNotFoundError:
        messagebox.showerror("Error", f"No se encontró la imagen: {background_image_path}")
        return
        

    # Etiqueta de bienvenida en el dashboard
    label_bienvenida = tk.Label(dashboard, text="¡Bienvenido al Dashboard!", font=("Arial", 20, "bold"), bg="#34495E", fg="white")
    label_bienvenida.pack(pady=120)
    
    # Botón para ejecutar el archivo de captura de imágenes
    boton_opcion1 = tk.Button(dashboard, text="Ejecutar captura de imágenes", font=("Arial", 14), width=30, bg="#34495E", fg="white", command=ejecutar_archivo_captura)
    boton_opcion1.pack(pady=10)
    
    # Botón para ejecutar el archivo de modelo
    boton_opcion2 = tk.Button(dashboard, text="Entrenar modelo", font=("Arial", 14), width=30, bg="#34495E", fg="white", command=ejecutar_archivo_modelo)
    boton_opcion2.pack(pady=10)
    
    # Botón para entrenar el detector de anomalías
    boton_opcion4 = tk.Button(dashboard, text="Entrenar Detector de Anomalias", font=("Arial", 14), width=30, bg="#34495E", fg="white", command=ejecutar_archivo_encoder)
    boton_opcion4.pack(pady=10)
    
    # Botón para ejecutar el archivo de reconocimiento de objetos
    boton_opcion3 = tk.Button(dashboard, text="Reconocimiento de objetos", font=("Arial", 14), width=30, bg="#34495E", fg="white", command=ejecutar_archivo_reconocimiento)
    boton_opcion3.pack(pady=10)
    
    boton_reportes = tk.Button(dashboard, text="Ver Reportes", font=("Arial", 14), width=30, bg="#34495E", fg="white", command=mostrar_reporte_anomalias)
    boton_reportes.pack(pady=10)

    # Botón para limpiar capturas
    boton_limpiar_capturas = tk.Button(dashboard, text="Limpiar Capturas", font=("Arial", 14), width=30, bg="#E74C3C", fg="white", command=limpiar_capturas)
    boton_limpiar_capturas.pack(pady=10)

    # Botón para limpiar modelo de entrenamiento
    boton_limpiar_entrenamiento = tk.Button(dashboard, text="Limpiar Entrenamiento", font=("Arial", 14), width=30, bg="#E74C3C", fg="white", command=limpiar_entrenamiento)
    boton_limpiar_entrenamiento.pack(pady=10)

    # Botón para limpiar modelo de anomalías
    boton_limpiar_anomalias = tk.Button(dashboard, text="Limpiar Entrenamiento Anomalías", font=("Arial", 14), width=30, bg="#E74C3C", fg="white", command=limpiar_anomalias)
    boton_limpiar_anomalias.pack(pady=10)

    # Iniciar el bucle principal del dashboard
    dashboard.mainloop()

# Funciones de limpieza
def limpiar_capturas():
    carpeta = "imagenes"
    if os.path.exists(carpeta):
        try:
            shutil.rmtree(carpeta)
            messagebox.showinfo("Éxito", "La carpeta 'imagenes' ha sido eliminada correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar la carpeta: {e}")
    else:
        messagebox.showinfo("Información", "La carpeta 'imagenes' no existe.")

def limpiar_entrenamiento():
    archivo = "modelo_objeto.h5"
    if os.path.exists(archivo):
        try:
            os.remove(archivo)
            messagebox.showinfo("Éxito", "El archivo 'modelo_objeto.h5' ha sido eliminado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar el archivo: {e}")
    else:
        messagebox.showinfo("Información", "El archivo 'modelo_objeto.h5' no existe.")
        
def mostrar_reporte_anomalias():
    archivo_csv = "detections_log.csv"
    if not os.path.exists(archivo_csv):
        messagebox.showerror("Error", f"El archivo {archivo_csv} no existe.")
        return

    try:
        # Leer el archivo CSV
        df = pd.read_csv(archivo_csv)

        # Verificar si la columna 'Error' existe
        if 'Camera ID' not in df.columns:
            messagebox.showerror("Error", "El archivo CSV no contiene la columna 'Camera ID'.")
            return

        # Contar los errores
        resumen_errores = df['Camera ID'].value_counts()

        # Crear gráfico de barras
        plt.figure(figsize=(8, 6))
        barras = resumen_errores.plot(kind='bar', color='orange')

        plt.title("Reporte de Anomalías", fontsize=16)
        plt.xlabel("Camara", fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Agregar totales encima de cada barra
        for i, total in enumerate(resumen_errores):
            plt.text(i, total + 0.5, str(total), ha='center', fontsize=10, color='black')

        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo generar el reporte: {e}")

def limpiar_anomalias():
    archivo = "autoencoder_model.h5"
    if os.path.exists(archivo):
        try:
            os.remove(archivo)
            messagebox.showinfo("Éxito", "El archivo 'autoencoder_model.h5' ha sido eliminado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo eliminar el archivo: {e}")
    else:
        messagebox.showinfo("Información", "El archivo 'autoencoder_model.h5' no existe.")



# Función para ejecutar el archivo de captura de imágenes y abrir CMD
def ejecutar_archivo_captura():
    try:
        ruta_archivo_captura = r"capturar_imagenes.py"
        
        # Ejecuta el archivo en una nueva ventana del símbolo de sistema (CMD)
        subprocess.Popen(["python", ruta_archivo_captura], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al ejecutar el archivo de captura: {str(e)}")


def ejecutar_archivo_encoder():
    try:
        ruta_archivo_encoder = r"encoder.py"  # Cambia al nombre correcto del archivo si es diferente
        
        # Ejecuta el archivo en una nueva ventana del símbolo de sistema (CMD)
        subprocess.Popen(["python", ruta_archivo_encoder], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al ejecutar el archivo del encoder: {str(e)}")        

# Función para ejecutar el archivo de modelo y abrir CMD
def ejecutar_archivo_modelo():
    try:
        ruta_archivo_modelo = r"entrenar_modelo.py"
        
        # Ejecuta el archivo en una nueva ventana del símbolo de sistema (CMD)
        subprocess.Popen(["python", ruta_archivo_modelo], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al ejecutar el archivo de modelo: {str(e)}")

# Función para ejecutar el archivo de reconocimiento de objetos y abrir CMD
def ejecutar_archivo_reconocimiento():
    try:
        ruta_archivo_reconocimiento = r"reconocimiento_tiempo_real.py"  # Cambia el nombre del archivo aquí
        
        # Ejecuta el archivo en una nueva ventana del símbolo de sistema (CMD)
        subprocess.Popen(["python", ruta_archivo_reconocimiento], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al ejecutar el archivo de reconocimiento: {str(e)}")

# Crear la ventana principal de login
ventana = tk.Tk()
ventana.title("Login")
ventana.geometry("400x300")
ventana.configure(bg="#2C3E50")  # Color de fondo principal

# Estilos personalizados para placeholders
def crear_entry_placeholder(entry, placeholder, color_placeholder="gray"):
    entry.insert(0, placeholder)
    entry.config(fg=color_placeholder)
    
    def on_focus_in(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="black")
    
    def on_focus_out(event):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.config(fg=color_placeholder)
    
    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

# Frame central para el login
frame_login = tk.Frame(ventana, bg="#ECF0F1", bd=5, relief="groove")
frame_login.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=300, height=250)

# Título de la ventana de login
label_titulo = tk.Label(frame_login, text="Iniciar Sesión", font=("Arial", 16, "bold"), bg="#ECF0F1", fg="#34495E")
label_titulo.pack(pady=10)

# Campo de entrada para el usuario con diseño
entry_usuario = tk.Entry(frame_login, font=("Arial", 12), bd=2, relief="solid", justify="center")
crear_entry_placeholder(entry_usuario, "Usuario")
entry_usuario.pack(pady=10, padx=10, ipady=5)

# Campo de entrada para la contraseña con diseño
entry_contraseña = tk.Entry(frame_login, font=("Arial", 12), bd=2, relief="solid", justify="center", show="*")
crear_entry_placeholder(entry_contraseña, "Contraseña")
entry_contraseña.pack(pady=10, padx=10, ipady=5)

# Botón para iniciar sesión
boton_login = tk.Button(frame_login, text="Iniciar sesión", font=("Arial", 12, "bold"), bg="#3498DB", fg="white", 
                        bd=0, relief="flat", command=verificar_credenciales, cursor="hand2")
boton_login.pack(pady=15, ipadx=10, ipady=5)

# Iniciar el bucle principal de la ventana de login
ventana.mainloop()
