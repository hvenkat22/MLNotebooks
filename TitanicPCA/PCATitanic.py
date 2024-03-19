import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

titanic = pd.read_csv("titanic.csv")
titanic=titanic.dropna()

pca = PCA(n_components=2)

pca.fit(titanic.drop("Survived", axis=1))

transformed_data = pca.transform(titanic.drop("Survived", axis=1))

titanic["PC1"] = transformed_data[:, 0]
titanic["PC2"] = transformed_data[:, 1]

print(titanic.head())