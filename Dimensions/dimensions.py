import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

def reductionDeDimension(df, verbose = 0):  
  if verbose > 1:
    print("Display data prior to reduction")
    sns.pairplot(df, hue = "class_revenue", vars = ['age', 'pages', 'first_item_prize', 'News_click', 'gender_encoded', 'country_encoded', 'ReBuy_encoded'])
  
  if verbose > 0:
    print("Setting X and Y")
  X = df.drop(["class_revenue"], axis=1)
  y = df["class_revenue"]
  
  if verbose > 0:
    print("Reducing to 2 components")
  pca = PCA(n_components=2)
  X_reduced = pca.fit_transform(X)
  
  if verbose > 1:
    print("Printing variance info")
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    
  if verbose > 0:
    print("Reducing dataset Dimensions")
  PCAResult = pd.DataFrame(X_reduced, columns = [f"PCA-{i}" for i in range (1,3)])
  PCAResult["class_revenue"] = y
  
  if verbose > 1:
    sns.pairplot(PCAResult, hue = 'class_revenue', vars = ['PCA-1', 'PCA-2'])  
  
  if verbose > 0:
    print("Last verification before return")
    PCAResult.info()
    PCAResult.head()
  
  return PCAResult
