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
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

file_name = "haarcascade_frontalface_alt2.xml"
face_detector = cv.CascadeClassifier(cv.data.haarcascades + file_name)

# Dicionário de rótulos
label = {0: "Sem capacete", 1: "Com capacete"}

# Variáveis de Controle de Alerta
ultimo_salvamento = 0
intervalo_minimo = 5  # segundos entre alertas (E-MAILS)
tempo_confirma_violacao = 3.0 # Tempo que a violação deve durar (em segundos)
inicio_violacao = 0 # Timestamp do início da detecção "Sem capacete" (pred=0)

print("Sistema de detecção de EPI iniciado (CTRL+C para sair)")

while True:
    status, frame = cam.read()
    if not status:
        print("Não foi possível capturar frame da câmera.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Flag para saber se alguma face "Sem Capacete" (pred=0) foi detectada neste frame
    violacao_detectada_neste_frame = False

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

            # Marcar violação
            if pred == 0:
                violacao_detectada_neste_frame = True
    
    # ============================
    # 🚨 Lógica de Confirmação de Violação
    # ============================
    agora = time.time()
    
    if violacao_detectada_neste_frame:
        # 1. Se a violação é nova, marca o início
        if inicio_violacao == 0:
            inicio_violacao = agora
            
        # 2. Verifica se a violação durou o tempo de confirmação E se o intervalo mínimo passou
        tempo_decorrido = agora - inicio_violacao
        
        if (tempo_decorrido >= tempo_confirma_violacao) and \
           (agora - ultimo_salvamento > intervalo_minimo):
            
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

    time.sleep(0.1)  # pequena pausa para evitar uso excessivo de CPU

cam.release()
print("Sistema encerrado.")