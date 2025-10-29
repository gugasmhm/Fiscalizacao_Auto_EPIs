import cv2 as cv
import functions
import os
import numpy as np
import time

# Inicializa a webcam
cam = cv.VideoCapture(0)

# Carregar o classificador de face (usando caminho oficial do OpenCV)
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv.CascadeClassifier(cv.data.haarcascades + file_name)

# Carregar dataset preparado
dataframe = functions.load_dataframe()

# Dividir dados
X_train, X_test, y_train, y_test = functions.split_dataset(dataframe)

# Treinar PCA e modelo
pca = functions.pca_model(X_train, n_components=30)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
knn = functions.knn_model(X_train_pca, y_train)

# Avaliação (opcional)
functions.evaluate_model(knn, pca, X_test, y_test)

# Dicionário de rótulos
label = {0: "Sem capacete", 1: "Com capacete"}

# Controle de salvamento (para não logar várias vezes seguidas)
ultimo_salvamento = 0
intervalo_minimo = 5  # segundos entre salvamentos

print("✅ Sistema iniciado. Pressione Q para sair.")

while True:
    status, frame = cam.read()
    if not status:
        break

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Só processa se houver pelo menos uma face detectada
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            if w < 50 or h < 50:
                # ignora rostos muito pequenos (ruído)
                continue

            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv.resize(face_roi, functions.IMG_SIZE)
            vector = pca.transform([face_roi.flatten()])
            pred = knn.predict(vector)[0]
            classification = label[pred]

            # Desenhar caixa e texto
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv.putText(frame, classification, (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Somente salva se detectar "Sem capacete" e respeitar intervalo mínimo
            agora = time.time()
            if pred == 0 and (agora - ultimo_salvamento > intervalo_minimo):
                functions.salvar_registro(frame)
                ultimo_salvamento = agora

    # Se nenhuma face for detectada, apenas exibe o vídeo sem log
    cv.imshow("Monitoramento - EPI Capacete", frame)

cam.release()
cv.destroyAllWindows()