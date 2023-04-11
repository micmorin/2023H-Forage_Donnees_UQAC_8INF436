# Projet de groupe - 8INF436 - Forage des Données - UQAC – Hiver 2023

# Membres de l'équipe
- David LELIEVRE - LELD23050303
- Jules DELAMARE - DELJ25090208
- Amani SATOURI 
- Michael MORIN - MORM07039500

# Guide du projet
1. [Introduction](#réduction-sélection-de-dimensions)
2. [Préparation des données](#préparation-des-données)
3. [Réduction/ Sélection de dimensions](#réduction-sélection-de-dimensions)
4. [Modèles de Classification](#modèles-de-classification)
5. [Interface Utilisateur](#interface-utlisateur)

## Introduction
Pour notre projet, nous avons décidé d'utiliser un dataset provenant d'un site web d'achat en ligne. Après avoir préparé nos données, vous avons choisis de *réduire/sélectionné* les dimensions *XYZ*. Quant au choix des modèles de classification, nous nous sommes arrêté sur 3 classifieurs différents. Nous avons choisis en premier RandomForest car il est rapide et efficace. Ensuite, nous avons choisis EFDT, également rapide et efficace, mais qui permet de faire de la classification en ligne. Enfin, nous avons choisis un arbre de décision classique car il est facile à comprendre et à interpréter. Finalement, notre interface utilisateur a été établie avec le framework Flask puisque l'équipe était familière avec ce dernier, il permet une association rapide avec le reste du projet en python et l'interface web est facilement accessible par tous.

*Recommendations *

Il est recommandé d'utiliser un environement virtuel et d'installer les modules nécessaire avant l'exécution ou le développement. De plus, pour le développement, le gitignore est préparé pour VS Code et Python. Voici un rappel des commandes pour ces recommendations:
```
python -m venv .env
.venv/Scripts/activate.bat (Terminal)
ou
.venv/Scripts./Activate.ps1 (Powershell)
pip install -r REQUIREMENTS.txt
```
Si un module est ajouté, veuillez utiliser `pip freeze > REQUIREMENTS.txt`.

## Preparation des données

## Réduction/ Sélection de dimensions

## Modèles de Classification

Nous avons donc fait 3 modeles différents :

### Random Forest

Un random forest, optimisé avec une gridSearchCV. Avec train/test comme stratégie de validation. <br>
Les métriques utilisées sont :
- Précision
- F1
- Recall

Grâce à la gridSearchCV, nous avons pu trouver les meilleurs paramètres pour notre modèle et obtenir des résultats plus précis, entre 0,9 et 1 sur toutes les métriques.

### Decision Tree

Un simple arbre de décision, optimisé avec une gridSearchCV. Avec train/test comme stratégie de validation. <br>
Les métriques utilisées sont :
- Précision
- F1

### Extremely Fast Decision tree

Un extremely Fast Decision Treen avec k-fold comme stratégie de validation.
Les métriques utilisées sont :
- Temps
- F1
- Accuracy
- rappel

## Interface Utilisateur