from Classification.models import classificationEnTroisModele
from Dimensions.dimensions import reductionDeDimension
from Interface.UI import interfaceUtilisateur
from Preparation.donnees import preparationDesDonnees

if __name__ == "__main__":
  donnees =  preparationDesDonnees()
  donnees = reductionDeDimension(donnees)
  modele1, modele2, modele3 = classificationEnTroisModele(donnees)
  interfaceUtilisateur(modele1, modele2, modele3, donnees)
