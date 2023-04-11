from Classification.models import classificationEnTroisModele
from Dimensions.dimensions import reductionDeDimension
from Interface.UI import interfaceUtilisateur
from Preparation.donnees import preparationDesDonnees
from sys import argv

if __name__ == "__main__":
  verbose = 0
  for arg in argv:
    if arg == "verbose=1": verbose = 1
    elif arg == "verbose=2": verbose = 2
    else: verbose = 0
  
  print("\033[33mPreparation Des Donnees\033[0m")
  donnees = preparationDesDonnees(verbose)

  print("\033[33mReduction de Dimension\033[0m")
  donnees = reductionDeDimension(donnees, verbose)

  print("\033[33mClassification en Trois Modeles\033[0m")
  modele1, modele2, modele3 = classificationEnTroisModele(donnees, verbose)

  print("\033[33mDebut de l'interface utilisateur\033[0m")
  interfaceUtilisateur(modele1, modele2, modele3, verbose)
