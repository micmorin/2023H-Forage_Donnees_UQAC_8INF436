# Projet de groupe - 8INF436 - Forage des Données - UQAC – Hiver 2023

# Membres de l'équipe
- David LELIEVRE - LELD23050303
- Jules DELAMARE - DELJ25090208
- Amani SATOURI - SATA03620109
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
Pour la préparation des données, nous avons tout d’abord remarqué que certaines variables étaient du mauvais type (par exemple les variables ‘first_item_prize’ et ‘revenue’ étaient de type ‘object’ alors qu’elles sont censées être de type numérique). Nous avons donc changé le type de ces variables-là. Nous avons ensuite réalisé des fonctions qui permettent de repérer les caractères spéciaux tels que ‘ ?’ qui ne sont pas forcément considérés automatiquement comme des valeurs manquantes. Pour les traiter, nous avons fais une fonction qui remplace tout d’abord ces caractères-là par ‘NaN’ afin de pouvoir visualiser le nombre de valeurs manquantes et les traiter plus tard. Pour le remplacement des valeurs manquantes pour les variables de type numérique nous avons utilisé la stratégie de remplacement par la médiane avec ‘SimpleImputer’. Nous avons remarqué grâce à des visualisation des variables la présence potentiel d’outliers et de bruit. Pour pallier cela, nous avons remplacé les outliers pars les bornes inférieure et supérieure avec la méthode ‘clip’ de pandas. Les bornes inférieures ont été calculées à partir des quartiles Q1 et Q3 et de l’écart inter quartile (IQR). Les valeurs en dehors de l’IQR ont été remplacé par la borne inférieure ou supérieure selon le cas. On à ensuite standardiser les données et encodé le reste des données. Nous n’avons pas réalisé d’échantillonnage sur les données car les classes ne sont pas déséquilibré dans ce dataset.

## Réduction/ Sélection de dimensions
Nous avons choisi de faire une réduction de dimension en faisant une Analyse de Composante Principale. Nous avons obtenus à l’issu de l’ACP 2 composantes principales. 
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
