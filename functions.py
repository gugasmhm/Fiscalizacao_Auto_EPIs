import numpy as np
import cv2 as cv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
from datetime import datetime
import smtplib
from email.message import EmailMessage

# Tamanho fixo para todas as imagens (ajuste se quiser)
IMG_SIZE = (160, 160)


def load_dataframe():
    """
    Carrega imagens das pastas:
      imagens/helmeton  -> rótulo 1 (com capacete)
      imagens/helmetoff -> rótulo 0 (sem capacete)

    Converte para grayscale, redimensiona para IMG_SIZE e flatten.
    Retorna um DataFrame pandas com colunas ARQUIVO, ROTULO, ALVO, IMAGEM.
    """
    dados = {"ARQUIVO": [], "ROTULO": [], "ALVO": [], "IMAGEM": []}

    for pasta, rotulo, alvo in [
        (f"imagens{os.sep}helmeton", "Com capacete", 1),
        (f"imagens{os.sep}helmetoff", "Sem capacete", 0),
    ]:
        if not os.path.exists(pasta):
            print(f"Aviso: pasta {pasta} não encontrada, será ignorada.")
            continue
        for arquivo in os.listdir(pasta):
            caminho = os.path.join(pasta, arquivo)
            img = cv.imread(caminho)
            if img is None:
                print(f"Aviso: não foi possível ler '{caminho}', pulando.")
                continue
            try:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Aviso: erro ao converter '{caminho}' -> {e}")
                continue
            # redimensiona para tamanho homogêneo
            gray = cv.resize(gray, IMG_SIZE)
            dados["ARQUIVO"].append(caminho)
            dados["ROTULO"].append(rotulo)
            dados["ALVO"].append(alvo)
            dados["IMAGEM"].append(gray.flatten())

    return pd.DataFrame(dados)


def split_dataset(df, test_size=0.3, random_state=42):
    """
    Recebe dataframe retornado por load_dataframe.
    Retorna: X_train, X_test, y_train, y_test como np.ndarray (dtype=float).
    """
    X = list(df["IMAGEM"])
    y = list(df["ALVO"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    # converter para numpy arrays com dtype float (requisito para PCA)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, X_test, y_train, y_test


def pca_model(X_train, n_components=30):
    """
    Ajusta o PCA no conjunto de treino (X_train).
    X_train deve ser array 2D: (n_amostras, n_features)
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    return pca


def knn_model(X_train, y_train):
    """
    GridSearchCV com KNN. Retorna o modelo já treinado (melhor estimator).
    """
    warnings.filterwarnings("ignore")
    grid_params = {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }
    model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, pca, X_test, y_test):
    """
    Avalia modelo: aplica PCA nas X_test e imprime classification_report e acurácia.
    """
    X_test_pca = pca.transform(X_test)
    preds = model.predict(X_test_pca)
    acc = accuracy_score(y_test, preds)
    print("\n📊 --- Relatório de Classificação ---")
    print(classification_report(y_test, preds, target_names=["Sem capacete", "Com capacete"]))
    print(f"✅ Acurácia do modelo: {acc * 100:.2f}%")
    return acc


def salvar_registro(frame):
    """
    Salva o frame em registros/ com timestamp (por exemplo ao detectar violação).
    Retorna o caminho da imagem salva.
    """
    os.makedirs("registros", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    caminho = f"registros/violacao_{timestamp}.jpg"
    cv.imwrite(caminho, frame)
    print(f"⚠️ Imagem salva: {caminho}")
    return caminho


def enviar_email_alerta(caminho_imagem):
    """
    Envia um e-mail para projetoepiestacio@gmail.com com a imagem da violação em anexo.
    É necessário usar uma senha de app gerada no Google.
    """
    remetente = "projetoepiestacio@gmail.com"
    senha = "pddy tocq fryj xfgv"
    destinatario = "projetoepiestacio@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = "⚠️ Alerta: Pessoa sem capacete detectada"
    msg["From"] = remetente
    msg["To"] = destinatario
    msg.set_content("Foi detectada uma pessoa sem capacete. A imagem está em anexo.")

    try:
        with open(caminho_imagem, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(caminho_imagem))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(remetente, senha)
            smtp.send_message(msg)
            print(f"📧 E-mail enviado com sucesso para {destinatario} com anexo {os.path.basename(caminho_imagem)}.")
    except Exception as e:
        print(f"❌ Erro ao enviar e-mail: {e}")
