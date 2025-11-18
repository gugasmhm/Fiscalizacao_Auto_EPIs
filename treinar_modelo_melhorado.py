import os
import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV # Importação de GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# =============================
# ⚙️ Configurações
# =============================
IMG_SIZE = (160, 160)
DATASET_COM = "imagens/helmeton"  # pasta com imagens de capacete
DATASET_SEM = "imagens/helmetoff"  # pasta sem capacete
# Removemos N_COMPONENTS pois vamos usar n_components=0.95 no PCA

# =============================
# 🔍 Função de pré-processamento
# =============================
def preprocess_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    img = cv.resize(img, IMG_SIZE)

    # Melhorias (mantidas para consistência com o server_stream_epi.py)
    clahe = cv.createCLAHE(2.0, (8, 8))
    img = clahe.apply(img)
    img = cv.bilateralFilter(img, 9, 75, 75)
    img = img.astype("float32") / 255.0
    return img.flatten()

# =============================
# 📂 Carregar dataset
# =============================
X = []
y = []

for folder, label in [(DATASET_COM, 1), (DATASET_SEM, 0)]:
    if not os.path.exists(folder):
        print(f"Pasta não encontrada: {folder}")
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            try:
                X.append(preprocess_image(path))
                y.append(label)
            except Exception as e:
                print(f"Erro ao processar {path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total de imagens carregadas: {len(X)}")

# =============================
# 🔀 Dividir em treino e teste
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================
# 🔽 PCA Otimizado
# =============================
# n_components=0.95 garante que 95% da variância seja retida.
# whiten=True transforma os componentes para terem variância unitária, o que ajuda o SVM.
pca = PCA(n_components=0.95, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"PCA ajustado: {pca.n_components_} componentes usados para 95% de variância.")

# =============================
# 🤖 Treinar SVM com GridSearchCV
# =============================
print("\nIniciando Grid Search para otimização do SVM...")
# Valores de C e gamma para buscar o melhor modelo
grid_params = {
    "C": [1, 10, 100],
    "kernel": ["rbf"],
    "gamma": ["scale", "auto"]
}
# cv=5: Cross-validation com 5 folds
# verbose=2: Para ver o progresso
grid_search = GridSearchCV(SVC(), grid_params, cv=5, verbose=2)
grid_search.fit(X_train_pca, y_train)

# Usar o melhor estimador encontrado
modelo = grid_search.best_estimator_
print(f"Melhores parâmetros SVM encontrados: {grid_search.best_params_}")

# =============================
# ✅ Avaliar
# =============================
y_pred = modelo.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {acc * 100:.2f}%")

# =============================
# 💾 Salvar modelo e PCA
# =============================
joblib.dump(modelo, "modelo_epi.pkl")
joblib.dump(pca, "pca_epi.pkl")
print("\nModelo e PCA salvos com sucesso! Utilize-os no server_stream_epi.py")