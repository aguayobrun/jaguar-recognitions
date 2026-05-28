import cv2
import os
from datetime import datetime
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# ─── Configuración ────────────────────────────────────────────────
API_KEY    = ""
MODEL_ID   = "find-jaguar-detection-2-qgqqd/2"   # workspace/modelo/versión
CONFIANZA  = 0.75                   # umbral mínimo de confianza
CARPETA    = "capturas_jaguar"
CLASE_JAGUAR = "jaguar detection"              # nombre exacto de la clase en tu modelo
ultimo_guardado = 0
COOLDOWN = 3  # segundos entre cada guardado
# ──────────────────────────────────────────────────────────────────

os.makedirs(CARPETA, exist_ok=True)
load_dotenv()                              # ← agrega esto
API_KEY = os.getenv("API_KEY")
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY,
)

cap = cv2.VideoCapture(0)  # 0 = cámara por defecto de la laptop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # ← agrega esta
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # ← y esta
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara lista. Presiona Q para salir.")
frame_count = 0
guardados   = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Analizar cada 10 frames para no saturar la API
    if frame_count % 10 == 0:
        # Guardar frame temporal para enviarlo
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            resultado = client.infer(temp_path, model_id=MODEL_ID)
            predicciones = resultado.get("predictions", [])
            # Debug: ver todo lo que detecta
            for pred in predicciones:
                print(f"Clase: '{pred.get('class')}' | Confianza: {pred.get('confidence'):.2f}")

            for pred in predicciones:
                
                clase      = pred.get("class", "").lower()
                confianza  = pred.get("confidence", 0)

               # ── Filtrar SOLO jaguares con confianza suficiente ──
            jaguares = [p for p in predicciones 
            if p.get("class", "").lower() == CLASE_JAGUAR 
            and p.get("confidence", 0) >= CONFIANZA]

            if jaguares:
                 pred = max(jaguares, key=lambda p: p.get("confidence", 0))
                 confianza = pred.get("confidence", 0)
                 tiempo_actual = datetime.now().timestamp()
                 if tiempo_actual - ultimo_guardado >= COOLDOWN:  # ← solo guarda si pasaron 3 segundos
                     ultimo_guardado = tiempo_actual
                     guardados += 1
                     ts     = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                     nombre = f"{CARPETA}/jaguar_{ts}_conf{confianza:.2f}.jpg"

                     x, y = int(pred["x"]), int(pred["y"])
                     w, h = int(pred["width"]), int(pred["height"])
                     x1, y1 = x - w // 2, y - h // 2
                     x2, y2 = x + w // 2, y + h // 2
                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 50), 2)
                     etiqueta = f"Jaguar {confianza:.0%}"
                     cv2.putText(frame, etiqueta, (x1, y1 - 8),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 50), 2)
                     cv2.imwrite(nombre, frame)
                     print(f"🐆 Jaguar detectado! Confianza: {confianza:.0%} → {nombre}")
                
        except Exception as e:
            print(f"⚠️  Error en API: {e}")

    # Mostrar vista en tiempo real
    cv2.putText(frame, f"Guardados: {guardados}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Deteccion de Jaguar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
os.remove("temp_frame.jpg") if os.path.exists("temp_frame.jpg") else None
print(f"\n✅ Sesión terminada. Jaguares registrados: {guardados}")