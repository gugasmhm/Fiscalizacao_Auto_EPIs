import functions

# Carregar dataset
df = functions.load_dataframe()

# Dividir em treino e teste
X_train, X_test, y_train, y_test = functions.split_dataset(df)

# Treinar PCA e modelo
pca = functions.pca_model(X_train, n_components=80)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn = functions.train_model(X_train_pca, y_train, modelo="knn")

# Gerar relatório
functions.evaluate_model(knn, pca, X_test, y_test)
