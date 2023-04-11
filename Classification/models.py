#importation de la librairie random forest
from sklearn.ensemble import RandomForestClassifier
#Import extremly fast decision tree
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
#import train_test_split
from sklearn.model_selection import train_test_split
#Importation de la librairie pour la validation croisée et pour l'optimisation des hyperparamètres
from sklearn.model_selection import KFold, GridSearchCV
#Importation des métriques de validation
from sklearn.metrics import accuracy_score, f1_score, recall_score
import time


def classificationEnTroisModele(donnees, verbose = 0):

  X = donnees.drop('class_revenue', axis=1)
  y = donnees['class_revenue']

  randomForest = randomForest(X, y)

  return randomForest, '', ''


#Fonction pour le premier modele, random forest
def randomForest(dataX, dataY, verbose = 0):

  #On crée le premier modèle de classification
  randomForest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

  #On divise les données en deux parties, une pour l'entrainement et une pour le test
  X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

  #On entraine le modèle
  randomForest.fit(X_train, y_train)

  #On test le modèle
  randomForest.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  from sklearn.metrics import classification_report, confusion_matrix
  y_pred = randomForest.predict(X_test)

  if verbose > 1:
    print(confusion_matrix(y_test, y_pred))

  if verbose > 0:
    print(classification_report(y_test, y_pred))

  #Les résultats ne sont pas satisfaisants, on va donc essayer d'optimiser le modèle
  #On va donc utiliser la fonction GridSearchCV pour trouver les meilleurs paramètres
  from sklearn.model_selection import GridSearchCV
  #On crée une liste de dictionnaires contenant les paramètres à tester
  param_grid = [{'n_estimators': [10, 100, 200], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
  #On crée un objet GridSearchCV
  grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)
  #On entraine le modèle
  grid.fit(X_train, y_train)

  if verbose > 0:
    #On affiche les meilleurs paramètres
    print(grid.best_params_)
    #On affiche les meilleurs estimateurs
    print(grid.best_estimator_)

  #On crée un nouveau modèle avec les meilleurs paramètres
  randomForest = grid.best_estimator_

  #On entraine le modèle
  randomForest.fit(X_train, y_train)

  #On test le modèle
  randomForest.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  y_pred = randomForest.predict(X_test)

  if verbose > 1:
    print(confusion_matrix(y_test, y_pred))

  if verbose > 0:
    print(classification_report(y_test, y_pred))
  return randomForest

#Fonction pour le deuxième modèle, Extremely Fast Decision Tree
def extremelyFastDecisionTree(dataX, dataY):
  #On crée le deuxième modèle de classification
  edtf = ExtremelyFastDecisionTreeClassifier()

  print(validationCroisee(dataX, dataY, 10, edtf))
  
  return edtf

#Fonction pour faire la validation Croisée des données
def validationCroisee(dataX, dataY, k, dt = ExtremelyFastDecisionTreeClassifier()):
    kf = KFold(n_splits=k)

    #Tableau des différents résultats
    precision = []
    rappel = []
    f1 = []

    for train_index, test_index in kf.split(dataX):
        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test = dataX.iloc[train_index].values, dataX.iloc[test_index].values
        y_train, y_test = dataY.iloc[train_index].values, dataY.iloc[test_index].values
        
        # Entraînement du modèle sur l'ensemble d'entraînement
        start_time = time.time()
        dt.partial_fit(X_train, y_train)
        elapsedTime = time.time() - start_time #Le temps pris pour l'entrainement
        
        y_pred = dt.predict(X_test)
        
        scorePrecision = accuracy_score(y_test, y_pred)
        scoref1 = f1_score(y_test, y_pred, average='micro')
        scoreRapp = recall_score(y_test, y_pred, average='micro')
        
        f1.append(round(scoref1,2))
        precision.append(round(scorePrecision,2))
        rappel.append(round(scoreRapp,2))
    
    return {"f1":f1,"precision":precision,"rappel":rappel,"time":elapsedTime}

