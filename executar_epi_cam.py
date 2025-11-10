import cv2 as cv
import functions
import joblib
import time
import numpy as np

# Carregar modelo e PCA treinados
print("🔹 Carregando modelo salvo...")
modelo = joblib.load("modelo_epi.pkl")
pca = joblib.load("pca_epi.pkl")

# Inicializa a webcam
cam = cv.VideoCapture(0)
file_name = "haarcascade_frontalface_alt2.xml"
face_detector = cv.CascadeClassifier(cv.data.haarcascades + file_name)

# Dicionário de rótulos
label = {0: "Sem capacete", 1: "Com capacete"}

ultimo_salvamento = 0
intervalo_minimo = 5  # segundos entre alertas

print("🎥 Sistema de detecção de EPI iniciado (pressione Q para sair)")

while True:
    status, frame = cam.read()
    if not status:
        break

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            if w < 50 or h < 50:
                continue

            # Pré-processamento
            roi = gray[y:y+h, x:x+w]
            roi = cv.resize(roi, functions.IMG_SIZE)
            roi = cv.GaussianBlur(roi, (3,3), 0)
            roi = cv.equalizeHist(roi)
            roi = roi.astype("float32") / 255.0

            # Converter para vetor PCA
            vector = pca.transform([roi.flatten()])
            pred = modelo.predict(vector)[0]
            texto = label[pred]
            color = (0,255,0) if pred == 1 else (0,0,255)

            # Exibir resultado
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, texto, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Registrar se estiver sem capacete
            agora = time.time()
            if pred == 0 and (agora - ultimo_salvamento > intervalo_minimo):
                caminho = functions.salvar_registro(frame)
                functions.enviar_email_alerta(caminho)
                ultimo_salvamento = agora

    cv.imshow("Monitoramento de EPI - Capacete", frame)

cam.release()
cv.destroyAllWindows()
print("🛑 Sistema encerrado.")
