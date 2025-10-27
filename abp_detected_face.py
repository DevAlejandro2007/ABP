import cv2
import tkinter as tk
from tkinter import Canvas
import threading

# Variable global para almacenar cuántas caras se detectaron
face_detected = 0
running = True

def camera_on(update_callback):
    """Inicia la cámara y detecta rostros en tiempo real"""
    global face_detected, running

    cascpath = "haarcascade_frontalcatface.xml"
    plate = cv2.CascadeClassifier(cascpath)

    video_capture = cv2.VideoCapture(0)

    while running:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta rostros
        plates = plate.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Dibuja rectángulos en los rostros detectados
        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_detected += 1
            update_callback(face_detected)  # Actualiza el contador en la interfaz

        # Muestra la ventana de video
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    video_capture.release()
    cv2.destroyAllWindows()

def dibujo_neumorfico(canvas):
    """Dibuja el efecto neumórfico en el canvas"""
    # Capa clara
    canvas.create_oval(
        10, 10, 290, 140,
        fill="#e0e0e0",
        outline="",
        width=0
    )
    # Capa sombra para efecto 3D
    canvas.create_oval(
        5, 5, 295, 145,
        fill="#e0e0e0",
        outline="",
        width=0
    )

def estadistica():
    """Crea la ventana Tkinter con contador neumórfico"""
    wind = tk.Tk()
    wind.title("Contador Neumórfico")
    wind.geometry("400x250")
    wind.config(bg="#e0e0e0")

    # Canvas para el contador neumórfico
    canva = Canvas(
        wind,
        width=300,
        height=150,
        bg="#e0e0e0",
        highlightthickness=0,
        bd=0
    )
    canva.pack(pady=40)

    dibujo_neumorfico(canva)

    # Texto del contador
    texto_contador = canva.create_text(
        150, 75,
        text="0",
        fill="#333",
        font=("Helvetica", 32, "bold")
    )

    # Función para actualizar el contador dinámicamente
    def actualizar_contador(valor):
        canva.itemconfig(texto_contador, text=str(valor))

    # Hilo separado para no bloquear Tkinter
    threading.Thread(target=camera_on, args=(actualizar_contador,), daemon=True).start()

    wind.mainloop()

if __name__ == "__main__":
    estadistica()

