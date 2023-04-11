import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder

def preparationDesDonnees():

  def detect_missing_values_files(file, missing_values):
    df = pd.read_csv(file, na_values = missing_values)
    missing_value_counts = df.isnull().sum() 
    return df, missing_value_counts
  
  def func_loc_missing_values(file, missing_values):
    df, missing_value_counts = detect_missing_values_files(file, missing_values)
    missing_values_location = df[df.isnull().any(axis=1)].index
    return missing_values_location
  
  def missing_values_NaN(customer_clean, missing_values):
    customer_clean = pd.DataFrame(customer_clean)
    customer_clean.replace(to_replace=missing_values, value=np.nan, inplace=True)
    return df
  
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
  
  customer = pd.read_csv("Customer.csv")

  # On examine les données de customer
  customer.head() 
  customer.info()
  customer.describe(include = "all")

  # visualisation des valeurs numériques avec des histogrammes
  customer.hist(bins = 50, figsize = (20,15))
  plt.show()  

  # On initialise la liste qui contient plusieurs caractères qui représentent des données manquantes
  missing_values = ["NaN", "N/a", "na", "unknown", '?','-', np.nan]

  # Affichage du nombre de valeurs manquantes pour les variables du fichier Customer.csv
  df, missing_value_counts = detect_missing_values_files("Customer.csv", missing_values)
  print(missing_value_counts)

  # Fonction qui fait appel à la fonction précédente pour la lecture du fichier 
  missing_values_location = func_loc_missing_values("Customer.csv", missing_values)
  print(missing_values_location)

  # Initialisation de la liste qui des données manquantes
  missing_values = ["NaN", "N/a", "na", "unknown", '?', np.nan]

  customer_clean = customer.copy()

  customer_clean = missing_values_NaN(df, missing_values)

  # Transformateur pour la fonction missing_values_NaN
  missing_values_transformer = FunctionTransformer(missing_values_NaN, validate=False, kw_args={'missing_values': missing_values})
  
  # Les missing values ont bien étés remplacés par NaN
  liste_indexes = [2,5,112,1233,2003,6678,9834]
  customer_clean.loc[liste_indexes]

  # On change le type des variables
  colonnes = ['first_item_prize', 'revenue']
  customer_clean = func_change_dtype(customer_clean, colonnes) 
  
  # Les variables ont bien étés étés converties en numérique
  customer_clean.info()

  # On sépare les variables numériques des autres variables 
  v_cat = ['gender','country','ReBuy']
  customer_num = customer_clean.drop(columns=v_cat, axis=1)

  #remplacement des valeurs manquantes pour les variables de type numérique.  
  imputer = SimpleImputer(strategy='median')
  imputer.fit(customer_num)
  X_num= imputer.transform(customer_num)

  df_customer_num=pd.DataFrame(X_num, columns=customer_num.columns)

  customer_num_copy = df_customer_num.copy() 
  print(df_customer_num) 

  # Gestion des variables quantitatives
  encoder = OneHotEncoder()
  v_cat = customer_clean[['gender','country','ReBuy', 'revenue']]
  v_cat_1hot = encoder.fit_transform(v_cat)

  customer_num.median().values
  X = df_customer_num
  v_cat.info()

  dataset = df_customer_num
  dataset_clamp = clamp_replace_bruit(dataset, 0.25, 0.75)
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
  df.head()

  return df
