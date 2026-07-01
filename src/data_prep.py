"""Chargement, decoupage et pipeline de preparation des donnees."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    RAW_DATA_PATH,
    TARGET,
    LEAK_COL,
    ID_COL,
    CAT_COLS,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)


def load_data(path=RAW_DATA_PATH):
    """Charge le CSV et retire la colonne de leakage.

    Returns:
        DataFrame pret pour la modelisation (cible incluse).
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[LEAK_COL])
    return df


def split_data(df):
    """Decoupe le DataFrame en train / val / test stratifies (72 / 10 / 18).

    Deux coupes successives :
    1. le test (18%) est isole en premier — touche une seule fois, a la fin
    2. la validation est prelevee dans les 82% restants ; VAL_SIZE etant
       exprime en % du total, on le convertit en % du restant

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Coupe 1 : test mis de cote
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Coupe 2 : 0.10 du total = 0.10 / 0.82 du restant
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def recode_categories(X):
    """Regroupe les codes hors documentation en 'autres' (mapping fixe).

    Definie ici (et pas dans un notebook) pour que le pipeline sauvegarde
    en joblib retrouve la fonction a l'import lors du rechargement.
    """
    X = X.copy()
    # education_level : 1=etudes sup, 2=universite, 3=lycee, 4=autres ; 0/5/6 -> 4
    X["education_level"] = X["education_level"].replace({0: 4, 5: 4, 6: 4})
    # marital_status : 1=marie, 2=celibataire, 3=autres ; 0 -> 3
    X["marital_status"] = X["marital_status"].replace({0: 3})
    return X


def make_pipeline(model, columns):
    """Assemble le pipeline complet : recodage + preparation + modele.

    Args:
        model: estimateur sklearn place en dernier maillon.
        columns: colonnes de X (sert a lister les numeriques).

    Returns:
        Pipeline pret a etre fitte sur des donnees brutes.
    """
    num_cols = [c for c in columns if c not in CAT_COLS + [ID_COL]]

    # Chaque famille de colonnes recoit son traitement, id est ecarte (remainder="drop")
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("recode", FunctionTransformer(recode_categories)),
        ("prep", preprocess),
        ("clf", model),
    ])
