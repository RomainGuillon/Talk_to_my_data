"""Scoring du jeu de test avec le modele sauvegarde (tache 11).

Produit reports/scoring_test.csv : id, proba_default, label_pred.
label_pred = 1 si proba >= seuil (seuil calcule dans train.py, logique top K%).

Usage (depuis la racine du repo) : python -m src.infer
"""

import joblib
import pandas as pd

from src.config import MODEL_PATH, SCORING_PATH, ID_COL
from src.data_prep import load_data, split_data
from src.metrics import evaluer


def score_test():
    """Score le jeu de test et ecrit le fichier de scoring.

    Returns:
        DataFrame (id, proba_default, label_pred), trie par proba decroissante.
    """
    bundle = joblib.load(MODEL_PATH)
    model, seuil = bundle["model"], bundle["seuil"]

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    proba = model.predict_proba(X_test)[:, 1]
    scoring = pd.DataFrame({
        "id": X_test[ID_COL].astype(int),
        "proba_default": proba,
        "label_pred": (proba >= seuil).astype(int),
    }).sort_values("proba_default", ascending=False)

    # Controle : performance sur le test (informel, le protocole a ete fige avant)
    evaluer("test : modele sauvegarde", y_test, proba, seuil=seuil)

    scoring.to_csv(SCORING_PATH, index=False)
    print(f"\nFichier de scoring : {SCORING_PATH} ({len(scoring)} lignes, "
          f"{scoring['label_pred'].sum()} clients a relancer)")
    return scoring


if __name__ == "__main__":
    score_test()
