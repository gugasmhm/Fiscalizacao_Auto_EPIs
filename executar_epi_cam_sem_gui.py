import cv2 as cv
import functions
import joblib
import time

# Carregar modelo e PCA treinados
print("Carregando modelo salvo...")
modelo = joblib.load("modelo_epi.pkl")
pca = joblib.load("pca_epi.pkl")

# Inicializa a webcam com ajustes para qualidade
cam = cv.VideoCapture(0, cv.CAP_V4L2)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

file_name = "haarcascade_frontalface_alt2.xml"
face_detector = cv.CascadeClassifier(cv.data.haarcascades + file_name)

# Dicionário de rótulos
label = {0: "Sem capacete", 1: "Com capacete"}
ultimo_salvamento = 0
intervalo_minimo = 5  # segundos entre alertas

print("Sistema de detecção de EPI iniciado (CTRL+C para sair)")

while True:
    status, frame = cam.read()
    if not status:
        print("Não foi possível capturar frame da câmera.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            if w < 50 or h < 50:
                continue

            # Pré-processamento com melhorias
            roi = gray[y:y+h, x:x+w]
            roi = cv.resize(roi, functions.IMG_SIZE)

            # CLAHE para contraste adaptativo
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)

            # Filtro bilateral para reduzir ruído mantendo bordas
            roi = cv.bilateralFilter(roi, 9, 75, 75)

            # Normalização
            roi = roi.astype("float32") / 255.0

            # Converter para vetor PCA
            vector = pca.transform([roi.flatten()])
            pred = modelo.predict(vector)[0]
            texto = label[pred]

            # Desenhar quadrado e texto no frame
            cor = (0, 255, 0) if pred == 1 else (0, 0, 255)
            cv.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
            cv.putText(frame, texto, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

            # Registrar se estiver sem capacete
            agora = time.time()
            if pred == 0 and (agora - ultimo_salvamento > intervalo_minimo):
                caminho = functions.salvar_registro(frame)
                print(f"ALERTA: Pessoa sem capacete detectada! Imagem salva em {caminho}")
                functions.enviar_email_alerta(caminho)
                ultimo_salvamento = agora

    time.sleep(0.1)  # pequena pausa para evitar uso excessivo de CPU

cam.release()
print("Sistema encerrado.")