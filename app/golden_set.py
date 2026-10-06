"""Golden set : questions de validation de l'assistant (PDF tache 14).

Pour chaque question, la valeur attendue est calculee ici avec pandas, sans
passer par les agents. On pose ensuite la question au superviseur et on compare.

Quatre types de cas :
- "nombres"   : les nombres attendus doivent se trouver dans le resultat du code execute
- "texte"     : les mots attendus doivent se trouver dans la reponse ou le resultat
- "graphique" : un graphique doit avoir ete produit
- "refus"     : l'assistant doit repondre que c'est impossible (PDF tache 16)

Quand un cas a une cle "agent", on verifie aussi que le superviseur a bien
appele cet agent specialise.

Usage (depuis la racine du repo) : python -m app.golden_set
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from app.agents.agent import creer_superviseur, poser_question
from app.agents.tools import DF
from src.config import K_TOP, MODEL_PATH, TARGET
from src.data_prep import split_data
from src.metrics import recall_at_topk

# Ecart accepte par defaut entre la valeur attendue et celle de l'agent
# (un arrondi a une decimale est accepte)
TOLERANCE = 0.05

# --- Valeurs attendues sur les donnees, calculees hors agent -----------------
# Les taux sont en pourcentage, comme les prompts le demandent aux agents.
taux_par_sexe = DF.groupby("sex")[TARGET].mean() * 100
taux_par_statut = DF.groupby("marital_status")[TARGET].mean() * 100
age_moyen_par_cible = DF.groupby(TARGET)["age"].mean()
plafond_median_par_cible = DF.groupby(TARGET)["limit_balance"].median()
moins_de_30_ans = DF[DF["age"] < 30]

# --- Valeurs attendues sur le modele, recalculees ici a partir du fichier ----
# (sans utiliser les tables importance et performance que lisent les agents)
if not MODEL_PATH.exists():
    raise RuntimeError("Modele absent : lancer d'abord python -m src.train.")

paquet = joblib.load(MODEL_PATH)
modele = paquet["model"]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(DF)
probas_test = modele.predict_proba(X_test)[:, 1]

# Client du jeu de test le plus risque, et nombre de clients au-dessus du seuil
id_client_le_plus_risque = X_test["id"].iloc[probas_test.argmax()]
nb_clients_a_relancer = (probas_test >= paquet["seuil"]).sum()
pr_auc_test = average_precision_score(y_test, probas_test)
recall_top20_test = recall_at_topk(y_test, probas_test, K_TOP)

noms = modele.named_steps["prep"].get_feature_names_out()
coefficients = modele.named_steps["clf"].coef_[0]
# "num__pay_0" -> "pay_0"
variable_la_plus_importante = noms[np.abs(coefficients).argmax()].split("__")[1]

GOLDEN_SET = [
    # --- ProfilingAgent ---
    {
        "question": "Combien de clients contient le jeu de données ?",
        "type": "nombres",
        "attendu": [len(DF)],
        "agent": "ProfilingAgent",
    },
    {
        "question": "Quel est le taux de défaut global ?",
        "type": "nombres",
        "attendu": [DF[TARGET].mean() * 100],
        "agent": "ProfilingAgent",
    },
    {
        "question": "Quel est l'âge moyen des clients ?",
        "type": "nombres",
        "attendu": [DF["age"].mean()],
        "agent": "ProfilingAgent",
    },
    {
        "question": "Quel est l'âge minimum et l'âge maximum des clients ?",
        "type": "nombres",
        "attendu": [DF["age"].min(), DF["age"].max()],
        "agent": "ProfilingAgent",
    },
    {
        "question": "Quel est le plafond de crédit moyen ?",
        "type": "nombres",
        "attendu": [DF["limit_balance"].mean()],
        "agent": "ProfilingAgent",
    },
    {
        "question": "Trace l'histogramme de l'âge des clients.",
        "type": "graphique",
        "agent": "ProfilingAgent",
    },
    # --- Comptages : plusieurs agents peuvent repondre, on ne verifie que la valeur ---
    {
        "question": "Combien de clients ont strictement plus de 60 ans ?",
        "type": "nombres",
        "attendu": [(DF["age"] > 60).sum()],
    },
    {
        "question": "Combien de clients ont le niveau d'éducation université (code 2) ?",
        "type": "nombres",
        "attendu": [(DF["education_level"] == 2).sum()],
    },
    {
        "question": "Combien de clients ont au moins un mois de retard de paiement "
                    "sur le mois le plus récent ?",
        "type": "nombres",
        "attendu": [(DF["pay_0"] >= 1).sum()],
    },
    {
        "question": "Quel est le taux de défaut des clients de strictement moins de 30 ans ?",
        "type": "nombres",
        "attendu": [moins_de_30_ans[TARGET].mean() * 100],
    },
    # --- SegmentationAgent ---
    {
        "question": "Quel est le taux de défaut par sexe ?",
        "type": "nombres",
        "attendu": [taux_par_sexe[1], taux_par_sexe[2]],
        "agent": "SegmentationAgent",
    },
    {
        "question": "Quel est le taux de défaut par statut marital ?",
        "type": "nombres",
        "attendu": [taux_par_statut[1], taux_par_statut[2], taux_par_statut[3]],
        "agent": "SegmentationAgent",
    },
    # --- ComparisonAgent ---
    {
        "question": "Compare l'âge moyen des clients en défaut et des clients sans défaut.",
        "type": "nombres",
        "attendu": [age_moyen_par_cible[0], age_moyen_par_cible[1]],
        "agent": "ComparisonAgent",
    },
    {
        "question": "Compare le plafond de crédit médian des clients en défaut "
                    "et des clients sans défaut.",
        "type": "nombres",
        "attendu": [plafond_median_par_cible[0], plafond_median_par_cible[1]],
        "agent": "ComparisonAgent",
    },
    # --- ModelInsightAgent (scores entre 0 et 1 : tolerance plus fine) ---
    {
        "question": "Quelle est la PR-AUC du modèle sur le jeu de test ?",
        "type": "nombres",
        "attendu": [pr_auc_test],
        "tolerance": 0.001,
        "agent": "ModelInsightAgent",
    },
    {
        "question": "Quel est le recall du modèle sur les 20 % de clients les plus risqués "
                    "du jeu de test ?",
        "type": "nombres",
        "attendu": [recall_top20_test],
        "tolerance": 0.001,
        "agent": "ModelInsightAgent",
    },
    {
        "question": "Quelle est la variable la plus importante du modèle ?",
        "type": "texte",
        "attendu": [variable_la_plus_importante],
        "agent": "ModelInsightAgent",
    },
    {
        "question": "Quel est le client qui a la plus grande probabilité de défaut ?",
        "type": "nombres",
        "attendu": [id_client_le_plus_risque],
        "agent": "ModelInsightAgent",
    },
    {
        "question": "Combien de clients du jeu de test le modèle propose-t-il de relancer ?",
        "type": "nombres",
        "attendu": [nb_clients_a_relancer],
        "agent": "ModelInsightAgent",
    },
    # --- Refus ---
    {
        "question": "Quel est le revenu moyen des clients ?",
        "type": "refus",
    },
    {
        "question": "Dans quelle ville habitent le plus de clients en défaut ?",
        "type": "refus",
    },
]


def extraire_nombres(resultat):
    """Met a plat tous les nombres d'un resultat de code.

    Le resultat peut etre un nombre, une Series, un DataFrame, une liste,
    un tuple ou un dictionnaire.

    Returns:
        Liste de nombres (vide si le resultat n'en contient pas).
    """
    if isinstance(resultat, pd.DataFrame):
        return list(resultat.select_dtypes("number").to_numpy().ravel())
    if isinstance(resultat, pd.Series):
        return list(pd.to_numeric(resultat, errors="coerce").dropna())
    if isinstance(resultat, dict):
        resultat = list(resultat.values())
    if isinstance(resultat, (list, tuple)):
        return [x for x in resultat if isinstance(x, (int, float, np.number))]
    if isinstance(resultat, (int, float, np.number)):
        return [resultat]
    return []


def verifier(cas, fiche):
    """Compare la reponse de l'assistant (fiche) a ce qui est attendu (cas).

    Returns:
        (ok, detail) : ok vaut True si le cas est valide, detail explique le verdict.
    """
    if cas["type"] == "refus":
        return fiche["refus"], f"refus = {fiche['refus']}"

    if fiche["refus"]:
        return False, "l'assistant a refuse alors qu'une reponse etait attendue"

    # Routage : le superviseur a-t-il appele l'agent attendu ?
    if "agent" in cas and cas["agent"] not in fiche["agents"]:
        return False, f"agent attendu {cas['agent']}, agents appeles {fiche['agents']}"

    if cas["type"] == "graphique":
        ok = any(execution["image"] is not None for execution in fiche["executions"])
        return ok, "graphique produit" if ok else "aucun graphique"

    if cas["type"] == "texte":
        # On cherche les mots attendus dans la reponse et dans les resultats du code
        texte = fiche["reponse"]
        for execution in fiche["executions"]:
            texte += " " + str(execution["resultat"])
        manquants = [mot for mot in cas["attendu"] if mot not in texte]
        if manquants:
            return False, f"mots absents : {manquants}"
        return True, f"mots retrouves : {cas['attendu']}"

    # Type "nombres" : tous les nombres produits par le code pour cette question
    nombres = []
    for execution in fiche["executions"]:
        nombres += extraire_nombres(execution["resultat"])

    tolerance = cas.get("tolerance", TOLERANCE)
    manquants = []
    for valeur in cas["attendu"]:
        trouve = any(abs(nombre - valeur) <= tolerance for nombre in nombres)
        if not trouve:
            manquants.append(round(float(valeur), 3))

    if manquants:
        obtenus = [round(float(nombre), 3) for nombre in nombres][:8]
        return False, f"attendu {manquants}, obtenu {obtenus}"
    return True, f"valeurs retrouvees : {[round(float(v), 3) for v in cas['attendu']]}"


if __name__ == "__main__":
    superviseur = creer_superviseur()

    nb_ok = 0
    for numero, cas in enumerate(GOLDEN_SET, start=1):
        fiche = poser_question(superviseur, cas["question"])
        ok, detail = verifier(cas, fiche)
        nb_ok += ok

        verdict = "OK   " if ok else "ECHEC"
        print(f"{numero:2d}. {verdict} | {cas['question']}")
        print(f"      agents : {fiche['agents']} | {detail}")
        if not ok:
            # En cas d'echec, on montre la reponse et le code pour comprendre
            print(f"      reponse : {fiche['reponse']}")
            for execution in fiche["executions"]:
                print(f"      code    : {execution['code']}")

    print(f"\nScore : {nb_ok} / {len(GOLDEN_SET)}")
