# Importações principais
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Carregar o dataset
iris = load_iris()
X = iris.data          # características (features) das flores
y = iris.target        # rótulos (0, 1, 2) para cada tipo de flor

# 2. Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Redução de dimensionalidade com PCA (para 2 dimensões, só para visualização)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. Definir o modelo e os hiperparâmetros a testar
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
knn = KNeighborsClassifier()

# 5. GridSearchCV para encontrar o melhor número de vizinhos (k)
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

# 6. Ver o melhor modelo encontrado
print("Melhor valor de k:", grid_search.best_params_)
print("Melhor score de validação:", grid_search.best_score_)

# 7. Avaliar no conjunto de teste
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print("Acurácia no teste:", acc)

# 8. (Opcional) Visualizar os dados transformados com PCA
plt.figure(figsize=(8, 6))
for label in set(y_train):
    plt.scatter(
        X_train_pca[y_train == label, 0],
        X_train_pca[y_train == label, 1],
        label=iris.target_names[label]
    )
plt.title("Dados Iris após PCA (2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True)
plt.show()
