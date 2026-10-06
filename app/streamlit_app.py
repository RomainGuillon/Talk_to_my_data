"""Application Streamlit du POC Talk to my Data (PDF tache 15).

Un champ pour poser une question, puis pour chaque reponse :
l'agent specialise appele par le superviseur, la reponse en francais,
le code Python execute et le resultat.
Les questions deja posees restent affichees dans l'historique.

Lancement (depuis la racine du repo) : streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit ne connait que le dossier app/ : on ajoute la racine du repo
# au chemin d'import pour retrouver les modules app.agents et src.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.agents.agent import creer_superviseur, poser_question

# Nombre maximal de lignes affichees pour un tableau (extrait du resultat)
MAX_LIGNES_AFFICHEES = 50

# Une question d'exemple par agent specialise
EXEMPLES = [
    "Trace l'histogramme de l'âge des clients.",
    "Quel est le taux de défaut par niveau d'éducation ?",
    "Compare le plafond de crédit moyen des clients en défaut et sans défaut.",
    "Quelles sont les 5 variables les plus importantes du modèle ?",
]

st.set_page_config(page_title="Talk to my Data", layout="wide")


@st.cache_resource
def charger_superviseur():
    """Cree le superviseur et ses agents une seule fois, puis les reutilise."""
    return creer_superviseur()


def afficher_resultat(execution):
    """Affiche le resultat d'une execution : tableau, valeur, texte ou graphique."""
    resultat = execution["resultat"]

    # Une Series devient un tableau a une colonne pour l'affichage
    if isinstance(resultat, pd.Series):
        resultat = resultat.to_frame()

    if isinstance(resultat, pd.DataFrame):
        st.dataframe(resultat.head(MAX_LIGNES_AFFICHEES))
        if len(resultat) > MAX_LIGNES_AFFICHEES:
            st.caption(f"{MAX_LIGNES_AFFICHEES} premières lignes sur {len(resultat)}.")
    elif resultat is not None:
        st.write(resultat)

    if execution["sortie"]:
        st.text(execution["sortie"])

    if execution["image"] is not None:
        st.image(execution["image"])


def afficher_fiche(fiche):
    """Affiche une reponse au format impose : agent, reponse, code execute, resultat."""
    if fiche["agents"]:
        st.caption("Agent appelé par le superviseur : " + ", ".join(fiche["agents"]))

    if fiche["refus"]:
        st.warning(fiche["reponse"])
    else:
        st.markdown(f"**Réponse :** {fiche['reponse']}")

    for execution in fiche["executions"]:
        st.markdown("**Code Python exécuté :**")
        st.code(execution["code"], language="python")
        st.markdown("**Résultat :**")
        afficher_resultat(execution)


# --- Etat de la session : l'historique des questions deja posees -----------
if "historique" not in st.session_state:
    st.session_state["historique"] = []

# --- Panneau lateral --------------------------------------------------------
with st.sidebar:
    st.header("Exemples de questions")
    for exemple in EXEMPLES:
        st.markdown(f"- {exemple}")

    st.header("Périmètre")
    st.write(
        "L'assistant répond à partir du jeu de données des clients carte de crédit "
        "et du modèle de scoring entraîné. Hors de ce périmètre, il répond : "
        "« Impossible avec les données disponibles. »"
    )

    if st.button("Effacer l'historique"):
        st.session_state["historique"] = []

# --- Page principale --------------------------------------------------------
st.title("Talk to my Data — défaut de carte de crédit")
st.write(
    "Posez une question en français sur les données. "
    "Chaque réponse affiche le code Python exécuté et son résultat."
)

# Le formulaire vide le champ apres l'envoi et valide avec la touche Entree
with st.form("formulaire_question", clear_on_submit=True):
    question = st.text_input("Votre question")
    envoyer = st.form_submit_button("Poser la question")

if envoyer and question.strip():
    try:
        with st.spinner("Analyse en cours…"):
            fiche = poser_question(charger_superviseur(), question.strip())
        # La reponse la plus recente est rangee en tete de l'historique
        st.session_state["historique"].insert(0, fiche)
    except Exception as erreur:
        st.error(f"La question n'a pas pu être traitée : {erreur}")

# --- Affichage : derniere reponse, puis historique --------------------------
historique = st.session_state["historique"]

if historique:
    derniere = historique[0]
    st.subheader(derniere["question"])
    afficher_fiche(derniere)

if len(historique) > 1:
    st.divider()
    st.subheader("Historique")
    for fiche in historique[1:]:
        with st.expander(fiche["question"]):
            afficher_fiche(fiche)
