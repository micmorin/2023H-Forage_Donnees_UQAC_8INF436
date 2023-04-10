import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
%matplotlib inline

def preparationDesDonnees():
  customer = pd.read_csv("Customer.csv")
  # On examine les données de customer
  customer.head() # On voit directement qu'il y a '?' pour first_item_prize à la 3e ligne
  # Il faudrait modifier le type de 'revenue' en une variable numérique
  #Il faudrait aussi modifier le type de 'first_item_prize' en une variable numérique
  # 10000 observations dans le tableau
  customer.info()
  customer.describe(include = "all")
  # statistiques pour les variables de type object
  customer.describe(include = object)
  # Il n'y a que 11 valeurs différentes pour la variable 'first_item_prize', anomalie possible
  # statistiques des variables numériques
  customer.describe() 
  # variables de type boolean
  customer.describe(include = bool)  
  
  # visualisation des valeurs numériques avec des histogrammes
  customer.hist(bins = 50, figsize = (20,15))
  plt.show()  
  # NOTE: Ici on traite les valeurs manquantes qui incluent notaùent 'unknown' ET '?' dans la meme fonction
  # Fonction qui permet de lire les valeurs manquantes pour les différents fichiers

  # On initialise la liste qui contient plusieurs caractères qui représentent des données manquantes
  missing_values = ["NaN", "N/a", "na", "unknown", '?','-', np.nan]
  def detect_missing_values_files(file, missing_values):
     # Lecture du fichier CSV en utilisant les valeurs manquantes issues de "missing_values"
    df = pd.read_csv(file, na_values = missing_values)
    # On compte les valeurs manquantes de chaque variable
    missing_value_counts = df.isnull().sum() 
    return df, missing_value_counts
  
# Affichage du nombre de valeurs manquantes pour les variables du fichier Customer.csv
  df, missing_value_counts = detect_missing_values_files("Customer.csv", missing_values)
  print(missing_value_counts)
# 4 missing values pour "revenue"
# 3 missing values pour"first_item_prize"  
# La fonction renvoie la localisation des valeurs manquantes dans revenue

# Fonction qui fait appel à la fonction précédente pour la lecture du fichier 
  def func_loc_missing_values(file, missing_values):
    df, missing_value_counts = detect_missing_values_files(file, missing_values)
    missing_values_location = df[df.isnull().any(axis=1)].index
    return missing_values_location
  missing_values_location = func_loc_missing_values("Customer.csv", missing_values)
  print(missing_values_location)
  
 # ON REMPLACE MAINTENANT LES VALEURS MANQUANTES DU DATASET PAR NaN POUR POUVOIR LES TRAITER PLUS TARD
 # La fonction missing_values_ va remplacer les valeurs manquantes par NaN pour qu'on puisse les manipuler par la suite
  def missing_values_NaN(customer_clean, missing_values):
    customer_clean =pd.DataFrame(customer_clean)
    customer_clean.replace(to_replace=missing_values, value=np.nan, inplace=True)
    return df

# Initialisation de la liste qui des données manquantes
  missing_values = ["NaN", "N/a", "na", "unknown", '?', np.nan]

# copie du fichier
  customer_clean = customer.copy()

# Appelle de fonction pour remplacer les valeurs manquantes par NaN
  customer_clean = missing_values_NaN(df, missing_values)

# Transformateur pour la fonction missing_values_NaN
  missing_values_transformer = FunctionTransformer(missing_values_NaN, validate=False, kw_args={'missing_values': missing_values})
# Les missing values ont bien étés remplacés par NaN
  liste_indexes = [2,5,112,1233,2003,6678,9834]
  customer_clean.loc[liste_indexes]
  
 # On change le type des variables
  def func_change_dtype(df, colonnes, dtype= float):
    df[colonnes] = df[colonnes].apply(pd.to_numeric, errors = 'coerce')
    df[colonnes] = df[colonnes].astype(dtype)
    return df
  colonnes = ['first_item_prize', 'revenue']
  customer_clean = func_change_dtype(customer_clean, colonnes) 
  # Les variables ont bien étés étés converties en numérique
  customer_clean.info()
  
  # On sépare les variables numériques des autres variables 
  #extraire que les varaibles_numériques
#faire un drop des caractéristiques de type objet et la variable cible
  v_cat = ['gender','country','ReBuy']
  customer_num = customer_clean.drop(columns=v_cat, axis=1)
  
#remplacement des valeurs manquantes pour les variables de type numérique.  
#de chaque caractéristique. la stratégie de remplacement est la mediane
  imputer = SimpleImputer(strategy='median')
#mantenant on peut appliquer l'instance SimpleImputer 
#au jeu de données -TBA- d'entrainement  en utilisant la méthode fit():
  imputer.fit(customer_num)
#transformer le jeu d’entrainement en
# remplaçant les valeurs manquantes par les médianes apprises
  X_num= imputer.transform(customer_num)
  df_customer_num=pd.DataFrame(X_num, columns=customer_num.columns)
  
  df_customer_num
#faire une copie du data numérique
  customer_num_copy = df_customer_num.copy() 
  print(df_customer_num) 
  
  # Gestion des variables quantitatives
  # Utilisation du transformateur OneHotEncoder
  encoder = OneHotEncoder()
  v_cat = customer_clean[['gender','country','ReBuy', 'revenue']]
  v_cat_1hot = encoder.fit_transform(v_cat)
  v_cat_1hot
  #verification
  customer_num.median().values
  X = df_customer_num
  v_cat.info()
  
  def clamp_replace_bruit(dataset, quartile_inf, quartile_sup):
    #Calcul du 1er quartile
    Q1 = dataset.quantile(quartile_inf)
    #calucl du 3e quartile
    Q3 = dataset.quantile(quartile_sup)
    # Calcul de l'interval interquartile
    iqr = Q3 - Q1
    #limite supérieure
    limite_sup = Q3 + 1.5 * iqr
    #limite inférieure
    limite_inf = Q1 - 1.5 * iqr
    # On remplace les valeurs jugées outliers, c'est à dire qui ne sont pas compris en la borne inf et la borne sup
    # On remplace les valeurs grace à clip soit par la borne inférieure, soit par la borne supérieure
    dataset = dataset.clip(lower = limite_inf, upper = limite_sup, axis = 1)
    return dataset

  dataset = df_customer_num
  dataset_clamp = clamp_replace_bruit(dataset, 0.25, 0.75)
  print(dataset_clamp) 
  
  #matrice de corrélation
  corr_matrix = X.corr()
  corr_matrix
  #On observe que les variables sont très faiblement corrélées entre elles
  # Il n'y a donc PAS besoin de plus étudier les variables 2 à 2 car on le fait uniquement si la corrélation est forte entre
  # 2 variables ce qui risquerait de biaiser le modèle ce qui n'est pas le cas ici
  
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
  # Fonction qui transforme la variable cible 'revenue' en format binaire 
  # Si observation de la variable 'revenue' < moyenne revenue = 0 sinon = 1
  def revenue_bin(dataset):
    moy_rev = dataset["revenue"].mean()
    dataset["class_revenue"] = (dataset["revenue"] > moy_rev).astype(int)
    return dataset.drop("revenue", axis = 1)
  df = revenue_bin(dataset=df)
  df.head()
  
  
  # Standardisation des données

# variables à standardiser
  cols_to_scale = ['age', 'pages', 'first_item_prize', 'News_click']
  scaler = StandardScaler()
  le = LabelEncoder()

# On encode les variables 'gender', 'country' et 'ReBuy'
  df['gender_encoded'] = le.fit_transform(df['gender'])
  df['country_encoded'] = le.fit_transform(df['country'])
  df['ReBuy_encoded'] = le.fit_transform(df['ReBuy'])

# On fit 
  df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# On supprime les colonnes orignales
  df =df.drop(['gender', 'country', 'ReBuy'], axis = 1)
  df.head()
  return df
