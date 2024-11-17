import cv2

def listar_camaras_disponibles():
    index = 0
    print("Buscando cámaras disponibles...")
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        else:
            print(f"Cámara detectada en índice: {index}")
            ret, frame = cap.read()
            if ret:
                print(f"Cámara {index} está funcionando correctamente.")
            else:
                print(f"Cámara {index} detectada, pero no está funcionando correctamente.")
        cap.release()
        index += 1

listar_camaras_disponibles()
