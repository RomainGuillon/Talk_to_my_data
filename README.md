# Talk to my Data — Scoring de défaut de crédit + POC GenAI

Projet de fin de formation **Data Scientist** — Datagong (Février 2026).

Contexte métier : banque de détail, direction Recouvrement & Risque. L'objectif est de prioriser les clients à relancer au prochain trimestre via un modèle de scoring, puis d'exposer les données à un assistant en langage naturel (POC GenAI).

Repo : https://github.com/RomainGuillon/Talk_to_my_data

---

## Structure du projet

```
/
├── README.md
├── requirements.txt
├── data/
│   └── raw/                         # CSV exporté depuis BigQuery (non commité)
├── notebooks/
│   ├── 01_setup_repo_et_eda.ipynb
│   ├── 02_modelisation_baseline.ipynb
│   └── 03_modelisation_advanced.ipynb
├── src/
│   ├── config.py                    # Chemins, constantes, noms de colonnes
│   ├── data_prep.py                 # Chargement, nettoyage, split, pipeline features
│   ├── metrics.py                   # PR-AUC, recall@topK, matrice de confusion
│   ├── train.py                     # Entraînement + sauvegarde modèle
│   └── infer.py                     # Scoring -> fichier (id, proba_default, label_pred)
├── app/
│   ├── streamlit_app.py
│   └── agents/
│       ├── prompts.py
│       ├── tools.py                 # Outil unique d'exécution Python contrôlée
│       └── agent.py
├── models/
│   └── model.joblib                 # Non commité
└── reports/
    ├── figures/
    └── model_report.md
```

---

## Dataset

Source : table BigQuery publique `bigquery-public-data.ml_datasets.credit_card_default`.

**Approche retenue** : export unique en CSV → déposé dans `data/raw/` → tout le projet lit le fichier local (pas de connexion BigQuery au runtime). Raison : contrainte no-réseau du POC + reproductibilité pour le jury.

| Propriété | Valeur |
|---|---|
| Lignes | 2 965 |
| Colonnes | 26 |
| Taux de défaut | ~22 % |
| Cible | `default_payment_next_month` (0/1) |

### Schéma des colonnes

| Colonne | Type | Description |
|---|---|---|
| `id` | float | Identifiant client |
| `limit_balance` | float | Plafond de crédit accordé |
| `sex` | int | Sexe (1=homme, 2=femme) |
| `education_level` | int | Niveau d'éducation |
| `marital_status` | int | Statut marital |
| `age` | float | Âge du client |
| `pay_0` à `pay_6` | float/int | Historique de remboursement (mois -1 à -6) |
| `bill_amt_1` à `bill_amt_6` | float | Montant du relevé mensuel |
| `pay_amt_1` à `pay_amt_6` | float | Montant payé chaque mois |
| `default_payment_next_month` | int | **Cible** — défaut le mois suivant (0/1) |
| `predicted_default_payment_next_month` | str | **Ne pas utiliser comme feature** (leakage) |

---

## Étapes du projet

**Étape 0 — Mise en place**
Récupération du dataset, `git init`, `.gitignore`, `requirements.txt`, README, environnement virtuel.

**Étape 1 — EDA** (`notebooks/01_setup_repo_et_eda.ipynb`)
Dictionnaire de données, contrôle qualité (valeurs manquantes, doublons, aberrations), exploration de la cible par segments, protocole d'évaluation.

**Étape 2 — Modèle de scoring** (`notebooks/02` et `03`, `src/`)
Pipeline sklearn sans leakage, baseline LogisticRegression, modèle avancé (HistGradientBoosting / XGBoost / LightGBM), choix du seuil aligné sur un coût métier, sauvegarde joblib, script d'inférence.

**Étape 3 — POC GenAI « Talk to my Data »** (`app/`)
Agent LangChain v1 avec un outil unique d'exécution Python contrôlée sur le DataFrame en mémoire. Contraintes : pas de SQL, pas de réseau, pas d'écriture disque. Chaque réponse affiche le code Python exécuté + le résultat. Interface Streamlit. Golden set de 10+ questions de validation.

**Étape 4 — Finalisation**
README complet, rapport synthétique, relecture, re-exécution propre, déploiement Cloud Run.

---

## Métriques retenues

- **PR-AUC** (prioritaire) — adapté au déséquilibre de classes
- **recall@topK** — logique métier : cibler les K% de clients les plus à risque
- Matrice de confusion, precision/recall au seuil retenu

Ne pas utiliser l'accuracy (trompeuse à ~22% de défaut).

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/RomainGuillon/Talk_to_my_data.git
cd Talk_to_my_data

# Créer et activer l'environnement virtuel
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Variable d'environnement (clé OpenAI)

Créer un fichier `.env` à la racine (jamais commité) :

```
OPENAI_API_KEY=sk-...
```

---

## Lancement

### Notebooks

```bash
jupyter notebook
```

Ouvrir dans l'ordre : `notebooks/01_...`, `02_...`, `03_...`.

### App Streamlit (POC GenAI)

Le modèle doit être entraîné au préalable (`src/train.py`) et le dataset présent dans `data/raw/`.

```bash
streamlit run app/streamlit_app.py
```

### Scoring sur le jeu de test

```bash
python src/infer.py
```

Produit `reports/scoring_test.csv` avec les colonnes `id`, `proba_default`, `label_pred`.

---

## Déploiement

L'application est déployée sur **Google Cloud Run** via Docker.

```bash
# Build de l'image
docker build -t talk-to-my-data .

# Push sur Artifact Registry puis déploiement Cloud Run
# (voir documentation GCP pour les détails)
```

La clé OpenAI est gérée via **Secret Manager GCP**, jamais dans l'image ni dans Git.

---

## Points de vigilance

- `predicted_default_payment_next_month` doit être droppée dès le chargement (leakage).
- Le split train/test est réalisé **avant** toute transformation (pipeline sklearn fitté sur le train uniquement).
- La clé API ne doit **jamais** être commitée.
- POC LangChain v1 : pas de SQL, pas d'accès réseau, pas d'écriture disque dans l'outil Python.

---

## Livrables (grille d'évaluation Datagong)

- Dépôt Git propre, reproductible, commits lisibles
- EDA : contrôles qualité, visualisations, compréhension de la cible
- Pipeline ML : pas de leakage, métriques adaptées au déséquilibre
- Modèle & décision : performance, choix de seuil/topK justifié
- POC GenAI : LangChain v1, affichage systématique du code, refus corrects
- Qualité générale : lisibilité, factorisation, commentaires utiles
