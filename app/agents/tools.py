"""Outil unique des agents : execution controlee de code pandas (PDF tache 12).

Le code ecrit par le modele est d'abord verifie (pas d'import, pas de lecture
ni d'ecriture de fichier, pas de reseau, pas de SQL), puis execute sur une copie
des tables en memoire. Le code doit ranger sa reponse dans la variable `result`.

Tables en memoire :
- df          : les clients (utilise par les agents Profiling, Segmentation, Comparison)
- importance  : poids des variables du modele sauvegarde (ModelInsightAgent)
- performance : scores du modele sur le jeu de test (ModelInsightAgent)
- scoring     : probabilite de defaut de chaque client du jeu de test (ModelInsightAgent)

Limite : c'est un controle par liste de regles, suffisant pour un POC.
Ce n'est pas un bac a sable de securite.

Test rapide (depuis la racine du repo) : python -m app.agents.tools
"""

import ast
import contextlib
import io

import matplotlib

matplotlib.use("Agg")  # rendu en memoire : aucune fenetre ne s'ouvre

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain.tools import tool
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.config import ID_COL, K_TOP, MODEL_PATH
from src.data_prep import load_data, split_data
from src.metrics import recall_at_topk

# DataFrame en memoire, charge une seule fois (colonne de leakage deja retiree)
DF = load_data()


def charger_tables_modele():
    """Construit trois tables sur le modele sauvegarde, sans le re-entrainer.

    Returns:
        (importance, performance, scoring), ou (None, None, None) si
        models/model.joblib n'existe pas.
        - importance  : une ligne par variable du modele, triee par importance.
                        Colonnes : variable, coefficient, importance (valeur absolue).
        - performance : une ligne par score mesure sur le jeu de test.
                        Colonnes : metrique, valeur.
        - scoring     : une ligne par client du jeu de test, triee du plus risque
                        au moins risque. Colonnes : id, proba_default, label_pred,
                        defaut_reel. Meme contenu que reports/scoring_test.csv,
                        plus le defaut reellement observe.
    """
    if not MODEL_PATH.exists():
        return None, None, None

    paquet = joblib.load(MODEL_PATH)
    modele = paquet["model"]
    seuil = paquet["seuil"]

    # Importance simple : coefficients de la regression logistique.
    # Les variables sont standardisees, donc les coefficients sont comparables.
    noms = modele.named_steps["prep"].get_feature_names_out()
    importance = pd.DataFrame({
        "variable": [nom.split("__")[1] for nom in noms],  # "num__age" -> "age"
        "coefficient": modele.named_steps["clf"].coef_[0],
    })
    importance["importance"] = importance["coefficient"].abs()
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)

    # Performance sur le jeu de test : meme decoupage que src/train.py
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(DF)
    probas = modele.predict_proba(X_test)[:, 1]
    predictions = (probas >= seuil).astype(int)
    performance = pd.DataFrame({
        "metrique": ["pr_auc", "recall_top_20", "precision", "recall",
                     "seuil", "nb_clients_test", "taux_defaut_test"],
        "valeur": [
            average_precision_score(y_test, probas),
            recall_at_topk(y_test, probas, K_TOP),
            precision_score(y_test, predictions),
            recall_score(y_test, predictions),
            seuil,
            len(y_test),
            y_test.mean(),
        ],
    })

    # Scoring : uniquement les clients du jeu de test, que le modele n'a jamais vus
    scoring = pd.DataFrame({
        "id": X_test[ID_COL].astype(int),
        "proba_default": probas,
        "label_pred": predictions,          # 1 = client a relancer (proba >= seuil)
        "defaut_reel": y_test,
    })
    scoring = scoring.sort_values("proba_default", ascending=False).reset_index(drop=True)

    return importance, performance, scoring


# Tables du modele, calculees une seule fois (None si le modele n'est pas entraine)
IMPORTANCE, PERFORMANCE, SCORING = charger_tables_modele()

# Taille maximale du texte renvoye au modele
MAX_LIGNES = 20
MAX_CARACTERES = 2000

# Seules fonctions Python de base visibles par le code execute
BUILTINS_AUTORISES = {
    "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "sorted": sorted, "range": range, "enumerate": enumerate, "zip": zip,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "print": print,
}

# Noms refuses : execution dynamique, fichiers, introspection
NOMS_INTERDITS = {
    "open", "exec", "eval", "compile", "input",
    "getattr", "setattr", "globals", "locals", "vars",
}

# Methodes refusees : ecriture disque, lecture de fichier ou d'URL, SQL
# (les methodes read_... sont refusees par leur prefixe, voir verifier_code)
METHODES_INTERDITES = {
    # query et eval executent du code ecrit dans une chaine de caracteres,
    # que verifier_code ne peut pas inspecter
    "query", "eval",
    # pandas
    "to_csv", "to_excel", "to_json", "to_html", "to_xml", "to_latex", "to_markdown",
    "to_parquet", "to_feather", "to_pickle", "to_hdf", "to_stata", "to_orc",
    "to_sql", "to_gbq", "to_clipboard",
    # numpy
    "save", "savez", "savez_compressed", "savetxt", "tofile",
    "load", "loadtxt", "genfromtxt", "fromfile", "memmap",
    # matplotlib
    "savefig", "imsave", "imread",
}

# Modules systeme refuses : pandas les importe en interne (ex. pd.io.common.os)
MODULES_INTERDITS = {
    "os", "sys", "subprocess", "shutil", "pathlib", "io", "builtins",
    "importlib", "socket", "urllib", "pickle",
}


def verifier_code(code):
    """Controle le code avant de l'executer, sans le lancer.

    ast.parse transforme le texte du code en arbre : chaque instruction,
    chaque nom et chaque appel de methode devient un noeud qu'on peut inspecter.

    Returns:
        None si le code est accepte, sinon le motif du refus (str).
    """
    try:
        arbre = ast.parse(code)
    except SyntaxError as e:
        return f"erreur de syntaxe : {e}"

    for noeud in ast.walk(arbre):
        # import os, from x import y
        if isinstance(noeud, (ast.Import, ast.ImportFrom)):
            return "les imports sont interdits (pd, np et plt sont deja disponibles)"

        # une boucle while peut ne jamais se terminer
        if isinstance(noeud, ast.While):
            return "les boucles while sont interdites"

        # un nom seul : open, eval, __builtins__...
        if isinstance(noeud, ast.Name):
            if noeud.id in NOMS_INTERDITS or noeud.id.startswith("__"):
                return f"nom interdit : {noeud.id}"

        # ce qui suit un point : df.to_csv, pd.read_csv, x.__class__...
        if isinstance(noeud, ast.Attribute):
            nom = noeud.attr
            if nom in METHODES_INTERDITES or nom.startswith("read_") or nom.startswith("__"):
                return f"methode interdite : {nom}"
            if nom in MODULES_INTERDITS:
                return f"module interdit : {nom}"

    return None


def figure_en_png(figure):
    """Convertit une figure matplotlib en image PNG gardee en memoire.

    Returns:
        Les octets de l'image (aucun fichier n'est ecrit).
    """
    tampon = io.BytesIO()
    figure.savefig(tampon, format="png", dpi=110, bbox_inches="tight")
    return tampon.getvalue()


def executer_code(code):
    """Verifie puis execute le code sur une copie du DataFrame.

    Returns:
        dict avec les cles :
        - "code"     : le code recu
        - "resultat" : valeur de la variable result (None si absente)
        - "image"    : graphique au format PNG (None si aucun graphique)
        - "sortie"   : texte affiche par print
        - "erreur"   : message d'erreur (None si tout s'est bien passe)
    """
    bilan = {"code": code, "resultat": None, "image": None, "sortie": "", "erreur": None}

    refus = verifier_code(code)
    if refus is not None:
        bilan["erreur"] = f"Code refuse : {refus}"
        return bilan

    # Espace de noms restreint : le code ne voit que ces objets.
    # Les tables sont des copies : les originaux ne peuvent pas etre modifies.
    espace = {
        "__builtins__": BUILTINS_AUTORISES,
        "df": DF.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    if IMPORTANCE is not None:
        espace["importance"] = IMPORTANCE.copy()
        espace["performance"] = PERFORMANCE.copy()
        espace["scoring"] = SCORING.copy()

    plt.close("all")  # on repart sans figure ouverte
    sortie = io.StringIO()
    try:
        with contextlib.redirect_stdout(sortie):
            exec(code, espace)
    except Exception as e:
        plt.close("all")
        bilan["erreur"] = f"{type(e).__name__} : {e}"
        return bilan

    bilan["sortie"] = sortie.getvalue()
    bilan["resultat"] = espace.get("result")

    # Si le code a trace un graphique, on le garde sous forme d'image
    if plt.get_fignums():
        bilan["image"] = figure_en_png(plt.gcf())
        plt.close("all")

    if bilan["resultat"] is None and bilan["image"] is None and bilan["sortie"] == "":
        bilan["erreur"] = "Le code n'a rien produit : ranger la reponse dans la variable result."
    return bilan


def resumer_bilan(bilan):
    """Texte renvoye au modele : l'erreur, ou un extrait du resultat."""
    if bilan["erreur"] is not None:
        return f"ERREUR. {bilan['erreur']}"

    morceaux = []
    resultat = bilan["resultat"]
    if isinstance(resultat, (pd.DataFrame, pd.Series)):
        morceaux.append(resultat.head(MAX_LIGNES).to_string())
        if len(resultat) > MAX_LIGNES:
            morceaux.append(f"({len(resultat)} lignes au total, {MAX_LIGNES} affichees)")
    elif resultat is not None:
        morceaux.append(str(resultat))
    if bilan["sortie"]:
        morceaux.append(bilan["sortie"])
    if bilan["image"] is not None:
        morceaux.append("Un graphique a ete cree.")

    return "\n".join(morceaux)[:MAX_CARACTERES]


@tool(response_format="content_and_artifact")
def executer_python(code: str):
    """Execute du code Python (pandas) sur les tables en memoire.

    Objets disponibles : df (clients carte de credit), importance, performance et
    scoring (tables du modele de scoring), pd (pandas), np (numpy), plt (matplotlib.pyplot).
    Le code doit ranger sa reponse dans une variable nommee `result`
    (nombre, texte, Series ou DataFrame). Pour un graphique, utiliser matplotlib
    sans appeler plt.show().
    Interdits : import, lecture ou ecriture de fichier, acces reseau, SQL,
    df.query et eval (filtrer avec df[condition]).

    Args:
        code: le code Python a executer

    Returns:
        Un extrait du resultat, ou un message d'erreur a corriger
    """
    bilan = executer_code(code)
    # 1er element : texte lu par le modele ; 2e element : bilan complet pour l'application
    return resumer_bilan(bilan), bilan


if __name__ == "__main__":
    # Trois codes acceptes, puis trois codes qui doivent etre refuses
    exemples = [
        "result = df['default_payment_next_month'].mean()",
        "result = df.groupby('sex')['default_payment_next_month'].mean()",
        "df['age'].plot.hist(bins=20)",
        "import os",
        "df.to_csv('copie.csv')",
        "result = pd.read_csv('https://exemple.com/donnees.csv')",
    ]
    for code in exemples:
        print(f">>> {code}")
        print(resumer_bilan(executer_code(code)))
        print()
