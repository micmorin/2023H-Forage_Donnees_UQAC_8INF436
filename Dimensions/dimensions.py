import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def preparationDeDimension(df):
   #Réduction de dimension : ACP 
    
  # On visualise les données avant la réduction de dimention
  sns.pairplot(df, hue = "class_revenue", vars = ['age', 'pages', 'first_item_prize', 'News_click', 'gender_encoded', 'country_encoded', 'ReBuy_encoded'])
  
  # On selectionne les données sans la variable cible
  X = df.drop(["class_revenue"], axis=1)
  
  # On selection la variable cible
  y = df["class_revenue"]
  
  # On réduit à 2 composants
  pca = PCA(n_components=2)
  X_reduced = pca.fit_transform(X)
  
  pca.explained_variance_
  pca.explained_variance_ratio_
  pca.singular_values_
  
  # On créer un DataFrame pour les composants principaux et pour la visualisation des données
  PCAResult = pd.DataFrame(X_reduced, columns = [f"PCA-{i}" for i in range (1,3)])
  PCAResult["class_revenue"] = y
  
  # Les données après la réduction de dimension
  sns.pairplot(PCAResult, hue = 'class_revenue', vars = ['PCA-1', 'PCA-2'])  
  
  PCAResult.info()
  PCAResult.head()
  
  return PCAResult
