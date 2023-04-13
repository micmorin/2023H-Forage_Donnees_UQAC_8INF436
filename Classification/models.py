from sys import stdout
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier

import time
import pandas as pd
import pickle
from os.path import isfile

def classificationEnTroisModele(donnees, verbose):

  X = donnees.drop('class_revenue', axis=1)
  y = donnees['class_revenue']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  if verbose > 1:
    print("\033[32mRandom forest\033[0m")

  if not isfile('./Classification/randomForestModel.pkl'):
    randomForestModel = randomForest(X_train, y_train, verbose)
    pickle.dump(randomForestModel, open('./Classification/randomForestModel.pkl', 'wb'))
  else:
    randomForestModel = pickle.load(open('./Classification/randomForestModel.pkl', 'rb'))

  if verbose > 1:
    print("\033[32mDecision Tree\033[0m")
  
  if not isfile('./Classification/decisionTreeModel.pkl'):
    decisionTreeModel = DecisionTree(X_train, y_train, verbose)
    pickle.dump(decisionTreeModel, open('./Classification/decisionTreeModel.pkl', 'wb'))
  else:
    decisionTreeModel = pickle.load(open('./Classification/decisionTreeModel.pkl', 'rb'))

  if verbose > 1:
    print("\033[32mExtremely Fast Decision Tree\033[0m")

  if not isfile('./Classification/extremelyFastDecisionTreeModel.pkl'):
    extremelyFastDecisionTreeModel = extremelyFastDecisionTree(X_train, y_train, 5, verbose)
    pickle.dump(extremelyFastDecisionTreeModel, open('./Classification/extremelyFastDecisionTreeModel.pkl', 'wb'))
  else:
    extremelyFastDecisionTreeModel = pickle.load(open('./Classification/extremelyFastDecisionTreeModel.pkl', 'rb')) 

  return randomForestModel, decisionTreeModel, extremelyFastDecisionTreeModel


#Fonction pour le premier modele, random forest
def randomForest(dataX, dataY, verbose = 0):

  X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

  #On crée le premier modèle de classification
  randomForest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
  randomForest.fit(X_train, y_train)
  randomForest.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  y_pred = randomForest.predict(X_test)

  if verbose > 1:
    print(confusion_matrix(y_test, y_pred))

  if verbose > 0:
    print(classification_report(y_test, y_pred))

  #On crée une liste de dictionnaires contenant les paramètres à tester
  param_grid = [{'n_estimators': [10, 100, 200], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
  
  #On crée un objet GridSearchCV
  grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=0)
  grid.fit(X_train, y_train)

  if verbose > 0:
    print(grid.best_params_)
    print(grid.best_estimator_)

  #On crée un nouveau modèle avec les meilleurs paramètres
  randomForest = grid.best_estimator_
  randomForest.fit(X_train, y_train)
  randomForest.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  y_pred = randomForest.predict(X_test)

  if verbose > 1:
    print(confusion_matrix(y_test, y_pred))

  if verbose > 0:
    print(classification_report(y_test, y_pred))
  return randomForest

#Fonction pour le deuxieme modèle, Decision Tree
def DecisionTree(dataX, dataY, verbose = 0):

  X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=42)

  #On crée le premier modèle de classification
  dt = DecisionTreeClassifier()
  dt.fit(X_train, y_train)
  dt.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  y_pred = dt.predict(X_test)

  if verbose > 1:
    print(confusion_matrix(y_test, y_pred))

  if verbose > 0:
    print(classification_report(y_test, y_pred))

  #On crée une liste de dictionnaires contenant les paramètres à tester
  param_grid = {
      'criterion': ['gini', 'entropy'],
      'max_depth': [None, 5, 10, 15],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
  }
  
  #On crée un objet GridSearchCV
  grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit=True, verbose=0)
  grid.fit(X_train, y_train)

  if verbose > 0:
    print(grid.best_params_)
    print(grid.best_estimator_)

  #On crée un nouveau modèle avec les meilleurs paramètres
  dt = grid.best_estimator_
  dt.fit(X_train, y_train)
  dt.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  y_pred = dt.predict(X_test)

  if verbose > 1:
    print(confusion_matrix(y_test, y_pred))

  if verbose > 0:
    print(classification_report(y_test, y_pred))

  return dt

#Fonction pour le troisième modèle, Extremely Fast Decision Tree
def extremelyFastDecisionTree(dataX, dataY, k, verbose = 0):

  edtf = ExtremelyFastDecisionTreeClassifier()

  print ("Valisation Croisee")
  #recherche de la meilleur coupe train/test
  search = validationCroisee(dataX, dataY, k, edtf, verbose)

  if verbose > 0:
    print(pd.DataFrame(search["scores"]))

  return edtf

#Fonction pour faire la validation Croisée des données
def validationCroisee(dataX, dataY, k, dt, verbose):
  kf = KFold(n_splits=k)

  #Tableau des différents résultats
  precision = []
  rappel = []
  f1 = []

  bestFitID = 0
  bestFitPrecision = 0 

  i=0
  for train_index, test_index in kf.split(dataX):
    stdout.write("Verifying folds ... " + str(i/5*100) + '%\r')
    stdout.flush()

    X_train, X_test = dataX.iloc[train_index].values, dataX.iloc[test_index].values
    y_train, y_test = dataY.iloc[train_index].values, dataY.iloc[test_index].values
    
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

    #check if the model is the best
    if scorePrecision > bestFitPrecision:
      bestFitID = i
      bestFitPrecision = scorePrecision
    i+=1
  
  return {"bestFitID":bestFitID,"scores":{"f1":f1,"precision":precision,"rappel":rappel,"time":elapsedTime}}
