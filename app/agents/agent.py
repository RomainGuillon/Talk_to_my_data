"""Systeme multi-agents Talk to my Data : un superviseur et quatre agents specialises.

Pattern "Tool Calling avec superviseur" (PDF taches 12, 13, 16) :
- chaque agent specialise (worker) est cree avec create_agent et utilise
  l'outil unique executer_python ;
- chaque worker est enveloppe dans un tool (demander_...) ;
- le superviseur est un agent dont les outils sont ces quatre tools.

Le format de sortie est impose par poser_question, qui renvoie toujours
les memes cles : la reponse en francais, le code execute et son resultat.

La cle OpenAI est lue dans .streamlit/secrets.toml (section [openai], cle api_key).

Test rapide (depuis la racine du repo) : python -m app.agents.agent
"""

import os
import tomllib

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from app.agents.prompts import (
    COMPARISON_PROMPT,
    MODEL_INSIGHT_PROMPT,
    PHRASE_REFUS,
    PROFILING_PROMPT,
    SEGMENTATION_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.agents.tools import IMPORTANCE, executer_python
from src.config import MAX_APPELS_AGENTS, MAX_APPELS_OUTIL, OPENAI_MODEL, SECRETS_PATH

# Les quatre agents specialises, ranges par nom. Rempli par creer_superviseur().
WORKERS = {}


def lire_cle_openai():
    """Lit la cle OpenAI.

    Ordre de recherche :
    1. la variable d'environnement OPENAI_API_KEY (cas d'un deploiement) ;
    2. le fichier .streamlit/secrets.toml, section [openai], cle api_key.

    Returns:
        La cle (str).
    """
    cle = os.getenv("OPENAI_API_KEY")
    if cle:
        return cle

    if not SECRETS_PATH.exists():
        raise RuntimeError(f"Fichier de secrets introuvable : {SECRETS_PATH}")

    # tomllib (livre avec Python) lit un fichier TOML et renvoie un dictionnaire
    with open(SECRETS_PATH, "rb") as fichier:
        secrets = tomllib.load(fichier)

    if "openai" not in secrets or "api_key" not in secrets["openai"]:
        raise RuntimeError(
            f"Cle absente de {SECRETS_PATH} : attendu une section [openai] avec api_key."
        )
    return secrets["openai"]["api_key"]


# --- Les workers ------------------------------------------------------------

def creer_worker(llm, prompt):
    """Cree un agent specialise : le modele, l'outil unique et son prompt.

    Returns:
        L'agent LangChain.
    """
    # Borne le nombre d'executions de code par question
    limite = ToolCallLimitMiddleware(
        tool_name="executer_python",
        run_limit=MAX_APPELS_OUTIL,
        exit_behavior="end",
    )
    return create_agent(
        model=llm,
        tools=[executer_python],
        system_prompt=prompt,
        middleware=[limite],
    )


def interroger_worker(nom, question):
    """Pose la question a un agent specialise et recupere ses executions de code.

    Returns:
        (reponse, trace) :
        - reponse : le texte de l'agent, lu par le superviseur
        - trace   : dict avec "agent" (son nom) et "executions" (liste des bilans
                    reussis de l'outil), garde pour l'application
    """
    etat = WORKERS[nom].invoke({"messages": [{"role": "user", "content": question}]})
    messages = etat["messages"]

    # Chaque appel de l'outil a laisse un ToolMessage, dont l'artifact est le bilan complet
    executions = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.artifact is not None:
            if message.artifact["erreur"] is None:
                executions.append(message.artifact)

    reponse = messages[-1].content
    return reponse, {"agent": nom, "executions": executions}


# --- Les workers enveloppes en tools pour le superviseur ---------------------

@tool(response_format="content_and_artifact")
def demander_profiling(question: str):
    """Decrit le jeu de donnees des clients : effectifs, taux de defaut global, moyenne,
    minimum, maximum, distribution ou histogramme d'une variable, comptages.

    Ce tool encapsule le ProfilingAgent.

    Args:
        question: la question de l'utilisateur, telle quelle

    Returns:
        La reponse du ProfilingAgent
    """
    return interroger_worker("ProfilingAgent", question)


@tool(response_format="content_and_artifact")
def demander_segmentation(question: str):
    """Calcule un taux de defaut par segment ou pour un groupe de clients
    (sexe, niveau d'education, statut marital, tranche d'age...).

    Ce tool encapsule le SegmentationAgent.

    Args:
        question: la question de l'utilisateur, telle quelle

    Returns:
        La reponse du SegmentationAgent
    """
    return interroger_worker("SegmentationAgent", question)


@tool(response_format="content_and_artifact")
def demander_comparaison(question: str):
    """Compare les clients en defaut et les clients sans defaut sur des variables
    numeriques : moyenne, mediane, boxplot, ecart.

    Ce tool encapsule le ComparisonAgent.

    Args:
        question: la question de l'utilisateur, telle quelle

    Returns:
        La reponse du ComparisonAgent
    """
    return interroger_worker("ComparisonAgent", question)


@tool(response_format="content_and_artifact")
def demander_modele(question: str):
    """Renseigne sur le modele de scoring deja entraine : variables utilisees,
    importance des variables, performance sur le jeu de test, probabilite de defaut
    predite par client, clients les plus risques, clients a relancer.

    Ce tool encapsule le ModelInsightAgent.

    Args:
        question: la question de l'utilisateur, telle quelle

    Returns:
        La reponse du ModelInsightAgent
    """
    if IMPORTANCE is None:
        message = "Le modele n'est pas disponible : lancer d'abord python -m src.train."
        return message, {"agent": "ModelInsightAgent", "executions": []}
    return interroger_worker("ModelInsightAgent", question)


# --- Le superviseur ---------------------------------------------------------

def creer_superviseur():
    """Cree les quatre agents specialises, puis le superviseur qui les orchestre.

    Returns:
        L'agent superviseur, pret a recevoir des questions.
    """
    # temperature=0 : meme question, meme code (utile pour le golden set)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=lire_cle_openai())

    WORKERS["ProfilingAgent"] = creer_worker(llm, PROFILING_PROMPT)
    WORKERS["SegmentationAgent"] = creer_worker(llm, SEGMENTATION_PROMPT)
    WORKERS["ComparisonAgent"] = creer_worker(llm, COMPARISON_PROMPT)
    WORKERS["ModelInsightAgent"] = creer_worker(llm, MODEL_INSIGHT_PROMPT)

    # Borne le nombre d'agents appeles par question (tous outils confondus)
    limite = ToolCallLimitMiddleware(run_limit=MAX_APPELS_AGENTS, exit_behavior="end")

    return create_agent(
        model=llm,
        tools=[demander_profiling, demander_segmentation, demander_comparaison, demander_modele],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[limite],
    )


def poser_question(superviseur, question):
    """Envoie une question au superviseur et range la reponse au format impose.

    Returns:
        dict avec les cles :
        - "question"   : la question posee
        - "reponse"    : la reponse finale en francais
        - "agents"     : noms des agents specialises appeles par le superviseur
        - "executions" : liste des executions de code reussies ; chaque element est le
                         bilan de l'outil (cles code, resultat, image, sortie, erreur)
        - "refus"      : True si l'assistant a repondu que c'est impossible
    """
    etat = superviseur.invoke({"messages": [{"role": "user", "content": question}]})
    messages = etat["messages"]

    # Chaque agent appele a laisse un ToolMessage, dont l'artifact est sa trace
    agents = []
    executions = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.artifact is not None:
            agents.append(message.artifact["agent"])
            executions += message.artifact["executions"]

    # Le dernier message est la reponse finale du superviseur
    reponse = messages[-1].content

    return {
        "question": question,
        "reponse": reponse,
        "agents": agents,
        "executions": executions,
        "refus": PHRASE_REFUS in reponse.lower(),
    }


if __name__ == "__main__":
    superviseur = creer_superviseur()

    # Une question par agent specialise, puis une question hors perimetre
    questions = [
        "Quel est l'âge moyen des clients ?",
        "Quel est le taux de défaut par sexe ?",
        "Compare le plafond de crédit moyen des clients en défaut et sans défaut.",
        "Quelles sont les 3 variables les plus importantes du modèle ?",
        "Quel est le revenu moyen des clients ?",
    ]
    for question in questions:
        fiche = poser_question(superviseur, question)
        print(f"QUESTION : {fiche['question']}")
        print(f"AGENTS   : {fiche['agents']}")
        print(f"REPONSE  : {fiche['reponse']}")
        for execution in fiche["executions"]:
            print("CODE :")
            print(execution["code"])
            print("RESULTAT :")
            print(execution["resultat"])
        print(f"REFUS    : {fiche['refus']}")
        print("-" * 60)
