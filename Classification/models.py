#importation de la librairie random forest
from sklearn.ensemble import RandomForestClassifier
#import train_test_split
from sklearn.model_selection import train_test_split

def classificationEnTroisModele(donnees):

  #On crée le premier modèle de classification
  modele1 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

  #On divise les données en deux parties, une pour l'entrainement et une pour le test
  X_train, X_test, y_train, y_test = train_test_split(donnees.drop('class_revenue', axis=1), donnees['class_revenue'], test_size=0.2, random_state=42)
  
  #On entraine le modèle
  modele1.fit(X_train, y_train)

  #On test le modèle
  modele1.score(X_test, y_test)

  #Evaluation du modèle via les métriques de validation adequates
  from sklearn.metrics import classification_report, confusion_matrix
  y_pred = modele1.predict(X_test)
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  


  return modele1, '', ''



