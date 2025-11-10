import functions
import joblib

print("🔧 Iniciando treinamento do modelo de EPI...")

# Carregar dataset
df = functions.load_dataframe()

# Dividir dados
X_train, X_test, y_train, y_test = functions.split_dataset(df)

# Treinar PCA e SVM
pca = functions.pca_model(X_train, n_components=80)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

modelo = functions.train_model(X_train_pca, y_train, modelo="svm")

# Avaliar modelo
functions.evaluate_model(modelo, pca, X_test, y_test)

# Salvar modelos
joblib.dump(modelo, "modelo_epi.pkl")
joblib.dump(pca, "pca_epi.pkl")
print("✅ Modelo e PCA salvos com sucesso!")
