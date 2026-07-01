"""Chemins et constantes du projet."""

from pathlib import Path

# Racine du repo, calculee depuis ce fichier (fonctionne depuis notebook ou script)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Chemins
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Credit_card_default.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
SCORING_PATH = REPORTS_DIR / "scoring_test.csv"

# Colonnes
TARGET = "default_payment_next_month"
LEAK_COL = "predicted_default_payment_next_month"  # a dropper, jamais en feature
ID_COL = "id"
CAT_COLS = ["sex", "education_level", "marital_status"]  # categorielles sans ordre

# Protocole d'evaluation (notebook 01, section 6.1)
TEST_SIZE = 0.18   # 18% test, touche une seule fois
VAL_SIZE = 0.10    # 10% validation, reglage hyperparametres
RANDOM_STATE = 42  # reproductibilite

# Decision metier (notebook 02, section 6) : budget de relance = 20% des clients
K_TOP = 0.20
