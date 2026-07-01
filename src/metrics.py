"""Metriques d'evaluation adaptees au desequilibre (notebook 02, section 3)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def recall_at_topk(y_true, y_proba, k=0.20):
    """Part des vrais defauts captes en ciblant les k% clients les plus risques."""
    n_top = int(len(y_true) * k)
    ordre = np.argsort(y_proba)[::-1]          # indices tries par proba decroissante
    y_top = np.asarray(y_true)[ordre[:n_top]]  # vraies etiquettes du top k%
    return y_top.sum() / np.asarray(y_true).sum()


def evaluer(nom, y_true, y_proba, seuil=0.5, k=0.20):
    """Affiche les metriques cles d'un modele. Retourne la PR-AUC."""
    y_pred = (y_proba >= seuil).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    print(f"--- {nom} ---")
    print(f"PR-AUC          : {pr_auc:.3f}  (aleatoire ~ {np.mean(y_true):.3f})")
    print(f"recall@top{int(k * 100):d}%    : {recall_at_topk(y_true, y_proba, k):.3f}")
    print(f"seuil {seuil:.3f} -> precision : {precision_score(y_true, y_pred):.3f}"
          f" | recall : {recall_score(y_true, y_pred):.3f}")
    print("matrice de confusion [[VN FP] [FN VP]] :")
    print(confusion_matrix(y_true, y_pred))
    return pr_auc
