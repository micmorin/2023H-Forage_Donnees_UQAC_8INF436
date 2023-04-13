import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sys import stdout
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder

def preparationDesDonnees(donnees, verbose = 0):

  def func_loc_missing_values(donnees):
    missing_values_location = donnees[donnees.isnull().any(axis=1)].index
    return missing_values_location
  
  def missing_values_NaN(donnees):
    donnees.replace(to_replace=["NaN", "N/a", "na", "unknown", '?','-', np.nan], value=np.nan, inplace=True)
    return donnees
  
  def func_change_dtype(df, colonnes, dtype= float):
    df[colonnes] = df[colonnes].apply(pd.to_numeric, errors = 'coerce')
    df[colonnes] = df[colonnes].astype(dtype)
    return df
  
  def clamp_replace_bruit(dataset, quartile_inf, quartile_sup):
    quartile_1 = dataset.quantile(quartile_inf)
    quartile_3 = dataset.quantile(quartile_sup)
    interQuartile = quartile_3 - quartile_1

    limite_sup = quartile_3 + 1.5 * interQuartile
    limite_inf = quartile_1 - 1.5 * interQuartile

    return dataset.clip(lower = limite_inf, upper = limite_sup, axis = 1)
  
  def revenue_bin(dataset):
    moy_rev = dataset["revenue"].mean()
    dataset["class_revenue"] = (dataset["revenue"] > moy_rev).astype(int)
    return dataset.drop("revenue", axis = 1) 

  if verbose > 0: 
    print("Examining data\n")
    donnees.head() 
    donnees.info()
    donnees.describe(include = "all")

  if verbose > 1: 
    print("\nVisualizing with Histograms")
    donnees.hist(bins = 50, figsize = (20,15))
    plt.show()  

  if verbose > 1:
    print("Display missing values locations")
    print(func_loc_missing_values(donnees))

  donnees = missing_values_NaN(donnees)

  # Transformateur pour la fonction missing_values_NaN
  missing_values_transformer = FunctionTransformer(missing_values_NaN, validate=False, kw_args={'missing_values': ["NaN", "N/a", "na", "unknown", '?','-', np.nan]})
  
  # Les missing values ont bien étés remplacés par NaN
  if verbose > 1:
    liste_indexes = [2,5,112,1233,2003,6678,9834]
    donnees.loc[liste_indexes]

  # On change le type des variables
  colonnes = ['first_item_prize', 'revenue']
  donnees = func_change_dtype(donnees, colonnes) 
  
  if verbose > 0: 
    print("Verify variables are all numerical")
    donnees.info()

  # On sépare les variables numériques des autres variables 
  v_cat = ['gender','country','ReBuy']
  customer_num = donnees.drop(columns=v_cat, axis=1)

  #remplacement des valeurs manquantes pour les variables de type numérique.  
  imputer = SimpleImputer(strategy='median')
  imputer.fit(customer_num)
  X_num= imputer.transform(customer_num)

  df_customer_num=pd.DataFrame(X_num, columns=customer_num.columns)

  # Gestion des variables quantitatives
  encoder = OneHotEncoder()
  v_cat = donnees[['gender','country','ReBuy', 'revenue']]
  v_cat_1hot = encoder.fit_transform(v_cat)

  customer_num.median().values
  X = df_customer_num

  if verbose > 0: 
    print("Display category information")
    v_cat.info()

  dataset = df_customer_num
  dataset_clamp = clamp_replace_bruit(dataset, 0.25, 0.75)

  if verbose > 0: 
    print("Display normalized dataset")
    print(dataset_clamp) 

  #matrice de corrélation
  corr_matrix = X.corr()

  # Enrichissement des données 
  dataset_cleaned = dataset_clamp.copy()
  dataset_cleaned['gender'] = v_cat['gender']
  dataset_cleaned['country'] = v_cat['country']
  dataset_cleaned['ReBuy'] = v_cat['ReBuy']

  # On positionne la variable cible dans la dernière colonne
  var_cible = 'revenue'
  cols = list(dataset_cleaned.columns)  
  cols.remove(var_cible)
  cols.append(var_cible)
  dataset_cleaned = dataset_cleaned[cols]
  
  # On reverifie que le dataset ne contient plus de valeurs manquantes
  dataset_cleaned.isnull().sum()
  df = dataset_cleaned.copy()

  # Numérisation des colonnes et fonction pour encoder 'revenue'
  df = revenue_bin(dataset=df)
  if verbose > 0: 
    print("Verify numerisation and encoding")
    df.head()

  # Standardisation des données
  cols_to_scale = ['age', 'pages', 'first_item_prize', 'News_click']
  scaler = StandardScaler()
  le = LabelEncoder()

  # On encode les variables 'gender', 'country' et 'ReBuy'
  df['gender_encoded'] = le.fit_transform(df['gender'])
  df['country_encoded'] = le.fit_transform(df['country'])
  df['ReBuy_encoded'] = le.fit_transform(df['ReBuy'])

  df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
  df =df.drop(['gender', 'country', 'ReBuy'], axis = 1)
  if verbose > 0: 
    print("Last check before returning data\n")
    df.head()

  return df