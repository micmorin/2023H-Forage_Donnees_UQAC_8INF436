from Prep import preparationDesDonnees
from Dim import preparationDeDimension
from Class import classificationEnTroisModele
from web import interfaceUtilisateur

if __name__ == "__main__":
  donnees =  preparationDesDonnees()
  donnees = preparationDeDimension(donnees)
  modele1, modele2, modele3 = classificationEnTroisModele(donnees)
  interfaceUtilisateur(modele1, modele2, modele3, donnees)
