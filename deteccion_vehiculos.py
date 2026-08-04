import cv2
import numpy as np
from pathlib import Path

class VehicleDetectorMotion:
    """Detector de vehículos usando detección de movimiento + tracking + semáforo"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.fgbg = None
        self.min_area = 1000
        self.kernel = None

        # 🔥 Línea virtual
        self.line_y = 400
        self.offset = 20

        # 🔥 Tracking
        self.vehicle_ids = 0
        self.tracked = []
        self.total_count = 0

    def load_model(self):
        print("🔄 Inicializando detector de movimiento...")

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=100,
            history=500
        )

        print("✓ Detector listo")
        return True

    def detect_vehicles(self, frame):
        fgmask = self.fgbg.apply(frame)

        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vehicles = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = w / float(h)

                if 0.5 < aspect_ratio < 5.0:
                    vehicles.append({
                        'box': (x, y, w, h),
                        'centroid': (x + w // 2, y + h // 2)
                    })

        return vehicles

    # 🔥 TRACKING + CONTEO REAL
    def update_tracking(self, vehicles):
        new_tracked = []

        for vehicle in vehicles:
            cx, cy = vehicle['centroid']
            matched = False

            for obj in self.tracked:
                ox, oy, oid, counted = obj

                if abs(cx - ox) < 50 and abs(cy - oy) < 50:
                    matched = True

                    # Cruce de línea
                    if not counted and abs(cy - self.line_y) < self.offset:
                        self.total_count += 1
                        counted = True

                    new_tracked.append((cx, cy, oid, counted))
                    break

            if not matched:
                new_tracked.append((cx, cy, self.vehicle_ids, False))
                self.vehicle_ids += 1

        self.tracked = new_tracked

    # 🔥 SEMÁFORO INTELIGENTE
    def traffic_light_logic(self, current_count):
        if current_count <= 5:
            return "VERDE", 20
        elif current_count <= 15:
            return "AMARILLO", 30
        else:
            return "ROJO", 40

    def draw_detections(self, frame, vehicles):
        for vehicle in vehicles:
            x, y, w, h = vehicle['box']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    def draw_line_and_info(self, frame):
        # Línea virtual
        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (255, 0, 0), 2)

        # Contador total
        cv2.putText(frame, f"Total vehiculos: {self.total_count}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        return frame

    def draw_traffic_light(self, frame, state):
        color_map = {
            "VERDE": (0,255,0),
            "AMARILLO": (0,255,255),
            "ROJO": (0,0,255)
        }

        cv2.putText(frame, f"Semaforo: {state}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color_map[state], 2)

        return frame


def run_camera():
    detector = VehicleDetectorMotion()

    if not detector.load_model():
        print("❌ Error")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo abrir cámara")
        return

    cap.set(3, 1280)
    cap.set(4, 720)

    frame_count = 0
    vehicles = []

    print("🚀 Sistema iniciado - presiona 'q' para salir")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 3 == 0:
            vehicles = detector.detect_vehicles(frame)

            # 🔥 tracking + conteo
            detector.update_tracking(vehicles)

            current_count = len(vehicles)

            # 🔥 semáforo
            state, _ = detector.traffic_light_logic(current_count)

            frame = detector.draw_detections(frame, vehicles)
            frame = detector.draw_line_and_info(frame)
            frame = detector.draw_traffic_light(frame, state)

        cv2.putText(frame, f"Vehiculos en frame: {len(vehicles)}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Sistema de Gestion Vial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
