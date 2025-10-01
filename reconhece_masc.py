import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Criar dados simulados
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'Temperatura': np.random.normal(30, 3, n),
    'Setor': np.random.choice(['Concreto', 'Elétrica', 'Hidráulica'], n),
    'Tarefa': np.random.choice(['Escavação', 'Instalação', 'Inspeção'], n),
    'Tempo_na_obra': np.random.randint(1, 30, n),
    'Supervisão': np.random.choice(['Sim', 'Não'], n),
    'Usando_Capacete': np.random.choice(['Sim', 'Não'], n, p=[0.7, 0.3])  # 70% usam
})

# 2. Separar features e rótulo
X = data.drop('Usando_Capacete', axis=1)
y = data['Usando_Capacete']

# 3. Pré-processamento
# Codificar variáveis categóricas
X_encoded = pd.get_dummies(X, drop_first=True)

# Padronizar (necessário para PCA e KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Codificar o rótulo
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Sim=1, Não=0

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# 5. Aplicar PCA para reduzir para 2 componentes
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 6. Treinar modelo com GridSearchCV para encontrar melhor K
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_pca, y_train)

# 7. Avaliação
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_pca)

print("Melhor valor de k:", grid_search.best_params_['n_neighbors'])
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Visualização
plt.figure(figsize=(8, 6))
colors = ['red' if label == 0 else 'green' for label in y_train]
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, alpha=0.6)
plt.title("Trabalhadores após PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
