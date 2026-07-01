# Rapport modele — Scoring defaut de credit

Etape 2 du projet (notebooks 02-03 + scripts src/train.py, src/infer.py).
Objectif metier : classer les clients par risque de defaut pour cibler un budget
de relance limite (~20% des clients du trimestre).

## Protocole

Split stratifie 72 / 10 / 18 (train / validation / test), fige au notebook 01.
Le test n'a servi a aucun choix (modele, seuil, K) et n'a ete touche qu'une fois
par notebook. Colonne de leakage `predicted_default_payment_next_month` droppee
des le chargement. Taux de defaut : ~21.4% — accuracy exclue, PR-AUC prioritaire.

## Modeles compares (validation)

| modele | PR-AUC | recall@top20% |
|---|---|---|
| **logreg balanced tunee C=0.01 (retenue, notebook 03)** | **0.655** | **0.594** |
| logreg balanced C=1 (baseline notebook 02) | 0.644 | 0.562 |
| hgb class_weight=balanced | 0.627 | 0.531 |
| hgb sans ponderation | 0.626 | 0.516 |

Le gradient boosting fait moins bien que la regression logistique : dataset petit
(2134 lignes de train), les modeles complexes n'ont pas assez de donnees.
La logreg `class_weight="balanced"` est retenue au notebook 02 ; le notebook 03
ajuste la regularisation (C=0.01 au lieu de 1 par defaut, via GridSearchCV en CV
5 plis sur le train), ce qui reduit le sur-apprentissage et ameliore toutes les
metriques de validation.

## Decision : top K% et seuil

- K = 20% impose par le budget de relance (contrainte metier, pas un optimum statistique).
- Seuil de probabilite equivalent (quantile 80% des probas de validation) : **0.537**.
  C'est lui qu'utiliserait un flux de production client par client.
- Analyse de sensibilite (relance 25 EUR, defaut manque 1000 EUR, notebook 02) :
  le cout brut diminue encore au-dela de 20% — argument pour discuter une hausse de
  capacite, mais ces couts supposent qu'une relance evite toujours le defaut (optimiste).

## Performance finale (test, 534 clients)

- PR-AUC : **0.498** (hasard ~0.213)
- recall@top20% : **0.456** — en relancant 20% des clients, on capte ~46% des defauts,
  soit 2.1x mieux que le hasard.
- Au seuil 0.537 : precision 0.488, recall 0.518 (121 clients flagges, ~23% :
  le seuil calibre sur la validation deborde legerement du budget de 20% sur le test).

L'ecart PR-AUC validation (0.655) / test (0.498) est notable : avec 297 lignes de
validation, les estimations sont instables. La valeur test est la reference.

## Enseignements du notebook 03

- Tuning : gain modeste et non confirme nettement sur le test (0.488 -> 0.498).
- Calibration : `class_weight="balanced"` sur-estime le risque (proba moyenne 0.449
  pour un taux reel de 0.215). Une calibration sigmoid/isotonic ramene les probas au
  bon niveau (Brier 0.189 -> ~0.12) sans changer le classement. Non integree au
  livrable (inutile pour le tri top K%), a activer si le metier lit les probas en absolu.
- Importance : `pay_0` (dernier statut de paiement) domine tres largement
  (permutation importance 0.30, ~0.01 pour les suivantes).

## Artefacts

- `models/model.joblib` : dict `{"model": pipeline sklearn complet, "seuil": 0.537}`.
  Le pipeline part des donnees brutes (recodage + one-hot + scaling inclus).
- `reports/scoring_test.csv` : id, proba_default, label_pred, trie par risque decroissant.
- Reproduction : `python -m src.train` puis `python -m src.infer` depuis la racine
  (les deux fichiers ci-dessus sont regeneres, ils ne sont pas commites).

## Limites et recommandations

- Dataset petit (2965 lignes) : ecarts val/test importants, tuning aux gains fragiles.
- Le test a ete utilise par les notebooks 02 puis 03 (documente) ; il ne sera plus touche.
- Probabilites non calibrees dans le livrable : suffisant pour un tri top K%.
- Hypotheses de cout (25 / 1000 EUR) a valider avec la direction Recouvrement.
- Recommandation : relancer les 20% clients les plus risques du scoring ; rediscuter
  le budget K avec la courbe de sensibilite du notebook 02, section 6.
