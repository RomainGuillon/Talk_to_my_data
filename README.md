# Talk to my Data — Scoring de défaut de crédit + POC GenAI

Projet de fin de formation **Data Scientist** — Datagong (Février 2026).

Contexte métier : banque de détail, direction Recouvrement & Risque. L'objectif est de prioriser les clients à relancer au prochain trimestre via un modèle de scoring, puis d'exposer les données à un assistant en langage naturel (POC GenAI).

Repo : https://github.com/RomainGuillon/Talk_to_my_data

---

## Structure du projet

```
/
├── README.md
├── pyproject.toml                   # Dépendances du projet (gérées avec uv)
├── uv.lock                          # Versions exactes installées
├── main.py                          # Entraînement puis scoring, en une commande
├── .streamlit/
│   └── secrets.toml.example         # Modèle du fichier de clé OpenAI (le vrai n'est pas commité)
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
│   ├── streamlit_app.py             # Interface : question, réponse, code, résultat, historique
│   ├── golden_set.py                # 21 questions de validation, comparées à des calculs pandas
│   └── agents/
│       ├── prompts.py               # Prompts des 4 agents spécialisés et du superviseur
│       ├── tools.py                 # Outil unique d'exécution Python contrôlée
│       └── agent.py                 # Création des agents et du superviseur
├── models/
│   └── model.joblib                 # Non commité
└── reports/
    ├── figures/
    ├── model_report.md
    └── scoring_test.csv             # Non commité
```

---

## Dataset

Source : table BigQuery publique `bigquery-public-data.ml_datasets.credit_card_default`.

**Approche retenue** : export unique en CSV → déposé dans `data/raw/` → tout le projet lit le fichier local (pas de connexion BigQuery au runtime). Raison : contrainte no-réseau du POC + reproductibilité pour le jury.

| Propriété | Valeur |
|---|---|
| Lignes | 2 965 |
| Colonnes | 26 |
| Taux de défaut | 21,4 % |
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
Récupération du dataset, `git init`, `.gitignore`, `pyproject.toml` (uv), README, environnement virtuel.

**Étape 1 — EDA** (`notebooks/01_setup_repo_et_eda.ipynb`)
Dictionnaire de données, contrôle qualité (valeurs manquantes, doublons, aberrations), exploration de la cible par segments, protocole d'évaluation. Les notebooks sont autonomes : ils n'importent rien depuis `src/`.

**Étape 2 — Modèle de scoring** (`notebooks/02` et `03`, `src/`)
Pipeline sklearn sans leakage, baseline LogisticRegression, comparaison avec HistGradientBoosting, réglage de la régularisation (modèle retenu : régression logistique `class_weight="balanced"`, `C=0.01`), choix du seuil aligné sur un coût métier, sauvegarde joblib, script d'inférence. Détail dans `reports/model_report.md`.

**Étape 3 — POC GenAI « Talk to my Data »** (`app/`)
Système multi-agents LangChain v1 : un superviseur et quatre agents spécialisés, qui partagent un outil unique d'exécution Python contrôlée sur des tables en mémoire. Contraintes : pas de SQL, pas de réseau, pas d'écriture disque. Chaque réponse affiche le code Python exécuté + le résultat. Interface Streamlit. Golden set de 21 questions de validation. Voir la section « POC GenAI » ci-dessous.

**Étape 4 — Finalisation**
README complet, rapport synthétique, relecture, re-exécution propre, déploiement Cloud Run (à faire).

---

## Métriques retenues

- **PR-AUC** (prioritaire) — adapté au déséquilibre de classes
- **recall@topK** — logique métier : cibler les K% de clients les plus à risque
- Matrice de confusion, precision/recall au seuil retenu

Ne pas utiliser l'accuracy (trompeuse à 21,4 % de défaut).

Résultats du modèle retenu sur le jeu de test (534 clients) : PR-AUC 0,498 (hasard : 0,213), recall@top20% 0,456. Détail et limites dans `reports/model_report.md`.

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/RomainGuillon/Talk_to_my_data.git
cd Talk_to_my_data

# Créer l'environnement virtuel et installer les dépendances (versions de uv.lock)
uv sync

# Activer l'environnement
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python 3.12. Les dépendances sont déclarées dans `pyproject.toml`.

### Clé OpenAI

Copier `.streamlit/secrets.toml.example` sous le nom `.streamlit/secrets.toml` (jamais commité), puis y mettre la clé :

```toml
[openai]
api_key = "sk-..."
```

La variable d'environnement `OPENAI_API_KEY`, si elle existe, est lue en priorité (cas d'un déploiement).

---

## Lancement

### Notebooks

```bash
jupyter notebook
```

Ouvrir dans l'ordre : `notebooks/01_...`, `02_...`, `03_...`.

Toutes les commandes suivantes se lancent depuis la racine du repo, environnement activé.

### Entraînement et scoring sur le jeu de test

```bash
python main.py
```

Équivaut à `python -m src.train` puis `python -m src.infer`. Produit `models/model.joblib` et `reports/scoring_test.csv` (colonnes `id`, `proba_default`, `label_pred`).

### App Streamlit (POC GenAI)

Le dataset doit être présent dans `data/raw/`, le modèle entraîné (commande ci-dessus) et la clé OpenAI renseignée.

```bash
streamlit run app/streamlit_app.py
```

### Validation du POC

```bash
python -m app.agents.tools     # outil seul, sans appel à OpenAI
python -m app.agents.agent     # cinq questions : une par agent, puis un refus
python -m app.golden_set       # 21 questions comparées à des calculs pandas
```

---

## POC GenAI — architecture

Pattern « Tool Calling avec superviseur » : le superviseur reçoit la question, choisit un agent spécialisé, puis rédige la réponse finale. Chaque agent spécialisé est enveloppé dans un tool du superviseur.

| Agent | Rôle | Tables utilisées |
|---|---|---|
| ProfilingAgent | Décrire les données : effectifs, moyennes, distributions, histogrammes | `df` |
| SegmentationAgent | Taux de défaut par segment, tableau trié, diagramme en barres | `df` |
| ComparisonAgent | Comparer clients en défaut et sans défaut : moyenne, médiane, écart | `df` |
| ModelInsightAgent | Variables du modèle, importance, performance sur le test, probabilité de défaut par client | `importance`, `performance`, `scoring` |

**Outil unique.** Les quatre agents utilisent le même outil, `executer_python`. Le code écrit par le modèle est d'abord contrôlé (pas d'import, pas de lecture ni d'écriture de fichier, pas de réseau, pas de SQL), puis exécuté sur des copies des tables en mémoire, avec un jeu réduit de fonctions autorisées.

**Tables en mémoire.** `df` contient les clients (colonne de leakage retirée). `importance`, `performance` et `scoring` sont calculées au démarrage à partir de `models/model.joblib`, sans ré-entraînement. `scoring` ne couvre que les 534 clients du jeu de test, que le modèle n'a jamais vus.

**Format de sortie.** Chaque réponse contient l'agent appelé, la réponse en français, le code Python réellement exécuté et son résultat (tableau, valeur ou graphique).

**Refus.** Si la question demande une information absente des données (revenu, ville, date...) ou sans rapport, l'assistant répond « Impossible avec les données disponibles. »

**Golden set.** `app/golden_set.py` pose 21 questions et compare les résultats à des valeurs calculées en pandas, sans passer par les agents. Il vérifie aussi l'agent choisi par le superviseur et les refus. Score obtenu le 6 octobre 2026 avec `gpt-4o-mini` : 21 sur 21.

**Limites.**
- Le contrôle du code est une liste de règles, suffisante pour un POC ; ce n'est pas un bac à sable de sécurité.
- La contrainte « pas d'accès réseau » porte sur le code exécuté par l'outil ; l'appel au modèle OpenAI passe par le réseau.
- Chaque question est traitée seule : l'assistant ne garde pas la mémoire de la question précédente.
- Le golden set valide ces 21 questions ; une autre formulation peut encore tromper l'assistant.

---

## Déploiement

Prévu à l'étape 4 (pas encore réalisé) : déploiement sur **Google Cloud Run** via Docker.

```bash
# Build de l'image
docker build -t talk-to-my-data .

# Push sur Artifact Registry puis déploiement Cloud Run
# (voir documentation GCP pour les détails)
```

La clé OpenAI sera gérée via **Secret Manager GCP** (variable d'environnement `OPENAI_API_KEY`), jamais dans l'image ni dans Git.

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
