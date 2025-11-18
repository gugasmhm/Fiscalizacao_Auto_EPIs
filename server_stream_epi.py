from flask import Flask, Response
import cv2 as cv
import time
import joblib
import functions

# ============================
# 🔧 Carregar modelo e PCA
# ============================
print("Carregando modelo salvo...")
modelo = joblib.load("modelo_epi.pkl")
pca = joblib.load("pca_epi.pkl")

# ============================
# 🎥 Inicializar câmera
# ============================
cam = cv.VideoCapture(0, cv.CAP_V4L2)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

# Detector de rosto
face_detector = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

# Rótulos
label = {0: "Sem capacete", 1: "Com capacete"}

# ============================
# 🚨 Variáveis de Controle de Alerta Otimizadas
# ============================
ultimo_salvamento = 0
intervalo_minimo_email = 10  # Aumentei para 10 segundos o intervalo MÍNIMO entre envios
tempo_confirma_violacao = 6.0 # Tempo que a violação deve durar (em segundos)
inicio_violacao = 0 # Timestamp do início da detecção "Sem capacete" (pred=0)

app = Flask(__name__)

# ============================
# 🔄 Função geradora do frame
# ============================
def gerar_frames():
    global ultimo_salvamento, inicio_violacao

    while True:
        status, frame = cam.read()
        if not status:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)
        
        # Flag para saber se alguma face "Sem Capacete" (pred=0) foi detectada neste frame
        violacao_detectada_neste_frame = False

        for (x, y, w, h) in faces:
            if w < 50 or h < 50:
                continue

            roi = gray[y:y+h, x:x+w]
            roi = cv.resize(roi, functions.IMG_SIZE)

            # Pré-processamento (deve ser idêntico ao do treino)
            clahe = cv.createCLAHE(2.0, (8, 8))
            roi = clahe.apply(roi)
            roi = cv.bilateralFilter(roi, 9, 75, 75)
            roi = roi.astype("float32") / 255.0

            vector = pca.transform([roi.flatten()])
            pred = modelo.predict(vector)[0]

            cor = (0, 255, 0) if pred == 1 else (0, 0, 255)
            texto = label[pred]

            cv.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv.putText(frame, texto, (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

            if pred == 0:
                violacao_detectada_neste_frame = True
                
        # ============================
        # 🚨 Lógica de Confirmação Otimizada
        # ============================
        agora = time.time()
        
        if violacao_detectada_neste_frame:
            # 1. Se a violação é nova, marca o início
            if inicio_violacao == 0:
                inicio_violacao = agora
                
            # 2. Verifica se a violação durou o tempo de confirmação E se o intervalo mínimo passou
            tempo_decorrido = agora - inicio_violacao
            
            if (tempo_decorrido >= tempo_confirma_violacao) and \
               (agora - ultimo_salvamento > intervalo_minimo_email):
                
                # A violação é real e persistente (>= 6s) e já passou o tempo mínimo para novo e-mail.
                caminho = functions.salvar_registro(frame)
                print(f"[ALERTA CONFIRMADO] Sem capacete por {tempo_decorrido:.2f}s! Imagem salva em: {caminho}")
                functions.enviar_email_alerta(caminho)
                
                # Resetar contadores
                ultimo_salvamento = agora
                inicio_violacao = 0
                
        else:
            # Se NENHUM rosto detectado neste frame está sem capacete, reseta o cronômetro
            inicio_violacao = 0


        # Converter frame para MJPEG
        ret, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# ============================
# 🌐 Rota de transmissão
# ============================
@app.route("/video")
def video():
    return Response(gerar_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================
# ▶ Iniciar servidor
# ============================
if __name__ == "__main__":
    print("Servidor iniciado: http://0.0.0.0:5000/video")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)