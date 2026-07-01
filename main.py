"""Point d'entree : entraine le modele puis produit le fichier de scoring.

Equivalent de : python -m src.train && python -m src.infer

Usage (depuis la racine du repo) : python main.py
"""

from src.train import train
from src.infer import score_test


def main():
    train()        # entraine et sauvegarde models/model.joblib
    print()
    score_test()   # recharge le modele et ecrit reports/scoring_test.csv


if __name__ == "__main__":
    main()
