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
  
  print("Preparation Des Donnees")
  donnees = preparationDesDonnees(verbose)

  print("Reduction de Dimension")
  donnees = reductionDeDimension(donnees, verbose)

  print("Classification en Trois Modeles")
  modele1, modele2, modele3 = classificationEnTroisModele(donnees, verbose)

  print("Debut de l'interface utilisateur")
  interfaceUtilisateur(modele1, modele2, modele3, verbose)
