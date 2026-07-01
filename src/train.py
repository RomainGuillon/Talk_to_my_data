"""Entrainement du modele retenu et sauvegarde joblib.

Modele : regression logistique class_weight='balanced' (notebook 02),
avec C=0.01 issu du tuning du notebook 03 (PR-AUC validation 0.644 -> 0.655).
Le seuil de decision est le quantile (1 - K_TOP) des probas de validation :
il correspond a relancer les K_TOP % clients les plus risques (budget metier).

Usage (depuis la racine du repo) : python -m src.train
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_PATH, RANDOM_STATE, K_TOP
from src.data_prep import load_data, split_data, make_pipeline
from src.metrics import evaluer


def train():
    """Entraine le pipeline, calcule le seuil et sauvegarde le tout.

    Returns:
        dict avec le pipeline fitte ('model') et le seuil de decision ('seuil').
    """
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    model = make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.01,
                           random_state=RANDOM_STATE),
        X_train.columns,
    )
    model.fit(X_train, y_train)

    # Seuil equivalent au top K% : quantile des probas de validation
    proba_val = model.predict_proba(X_val)[:, 1]
    seuil = float(np.quantile(proba_val, 1 - K_TOP))

    evaluer("validation : logreg class_weight=balanced", y_val, proba_val,
            seuil=seuil, k=K_TOP)

    bundle = {"model": model, "seuil": seuil}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModele sauvegarde : {MODEL_PATH} (seuil = {seuil:.3f})")
    return bundle


if __name__ == "__main__":
    train()
