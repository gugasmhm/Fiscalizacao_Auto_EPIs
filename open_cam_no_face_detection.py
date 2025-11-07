import cv2 as cv
import functions
import os
import numpy as np

cam = cv.VideoCapture(0)  # Iniciando a WebCam

# Carregar dataset preparado (imagens/helmeton e imagens/helmetoff)
dataframe = functions.load_dataframe()

# Dividir em treino e teste (retorna arrays)
X_train, X_test, y_train, y_test = functions.split_dataset(dataframe)

# Ajustar PCA (treinar) usando X_train
pca = functions.pca_model(X_train, n_components=30)

# Transformar os conjuntos para o espaço PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Treinar KNN no espaço PCA
knn = functions.knn_model(X_train_pca, y_train)

# Avaliar modelo (opcional)
functions.evaluate_model(knn, pca, X_test, y_test)

# Rótulo das classificações
label = {
    0: "Sem capacete",
    1: "Com capacete"
}

print("Sistema iniciado. Pressione Q para sair.")

# Abrindo a webcam...
while True:
    status, frame = cam.read()  # Lendo a imagem e extraindo frame
    if not status:
        break

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    height, width, _ = frame.shape
    # região central (ajuste o tamanho se necessário)
    size = 200
    pt1 = ((width // 2) - size // 2, (height // 2) - size // 2)
    pt2 = ((width // 2) + size // 2, (height // 2) + size // 2)
    region = frame[pt1[1]: pt2[1], pt1[0]: pt2[0]]

    # proteger contra frames pequenos
    if region.size == 0:
        continue

    gray_face = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    gray_face = cv.resize(gray_face, functions.IMG_SIZE)

    vector = pca.transform([gray_face.flatten()])  # Extraindo features da imagem
    pred = knn.predict(vector)[0]
    classification = label[pred]

    color = (0, 255, 0) if pred == 1 else (0, 0, 255)

    # Escrevendo classificação e desenhando retângulo
    cv.putText(frame, classification, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)
    cv.rectangle(frame, pt1, pt2, color, thickness=3)

    # Envia alerta se estiver sem capacete
    if pred == 0:
        caminho_img = functions.salvar_registro(frame)
        functions.enviar_email_alerta(caminho_img)

    cv.imshow("Cam - EPI Capacete (Região fixa)", frame)

cam.release()
cv.destroyAllWindows()

