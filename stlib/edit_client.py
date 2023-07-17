description = "Modification client"


def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import os

    import requests

    API_PREDICT_URL = os.environ.get("API_PREDICT_URL")
    RESUME_CAT_COL = os.environ.get("RESUME_CAT_COL")

    # Recupère la liste des colonnes catégorielles
    # et les versions OneHotEncodées reliées
    @st.cache_data
    def load_resume_cat_col():
        df = pd.read_pickle(RESUME_CAT_COL)
        return df

    @st.cache_data
    def predict_client(client_line):
        reponse = requests.get(
            API_PREDICT_URL, json={"data": client_line.to_json(default_handler=str)}
        )

        # Vérifier la réponse du serveur
        if reponse.status_code == 200:
            # Récupérer le DataFrame depuis la réponse JSON
            json_reponse = reponse.json()

            return json_reponse
        else:
            st.error("Erreur lors de la prédiction")

    # Vérifie si une valeur (texte) est un float
    def is_float(v):
        try:
            f = float(v)
        except ValueError:
            return False
        return True

    resume_cat_col = load_resume_cat_col()

    # Description de la page
    st.title(description)

    st.markdown(
        "Cette page offre la possibilité de modifier les caractéristiques du client afin de réaliser une simulation et de voir s'il pourrait être accepté s'il ne l'est pas déjà, ou s'il pourrait modifier certains termes de son contrat tout en restant éligible."
    )
    st.markdown(
        "Il est important de noter que cette prédiction est une simulation et ne prend pas en compte toutes les relations entre les différentes caractéristiques. Une véritable modification et simulation sera nécessaires pour valider définitivement ces changements."
    )

    with st.expander("Apercu des features"):
        st.image(
            shap_img,
            caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif",
        )

    # On récupère la liste des features
    options = list(client_line.index)
    # Et on retire toutes les OneHotEncoded binaires pour les remplacer par les catégorielles
    options = [c for c in options if not c.startswith(tuple(resume_cat_col["start"]))]
    options += list(resume_cat_col["simple"].unique())

    # On récupère le score brut
    old_score = predict_client(client_line.squeeze())["result_proba"]

    # On affiche la liste des features et on récupère la valeur d'origine
    # de celle selectionnée
    option = st.selectbox(
        "Feature à modifier", options, index=options.index("AMT_CREDIT")
    )
    old_value = client_line[option]
    st.caption("Valeur d'origine : " + str(old_value))

    # Liste des valeurs de la features
    unique_val = list(dataset[option].dropna().unique())

    # Si on est devant une variable catégorielle
    if option in list(resume_cat_col["simple"]):
        # On récupère le nom global de la feature
        ligne_option = resume_cat_col.loc[resume_cat_col["simple"] == option].iloc[0]
        start = ligne_option["start"] + "_"
        end = ligne_option["end"]

        # On récupère la liste des valeurs possibles
        list_option = list(
            resume_cat_col.loc[resume_cat_col["simple"] == option, "val"]
        )

        value = st.selectbox(
            "Valeur à lui donner", list_option, index=list_option.index(old_value)
        )

        # On met a jour les colonnes categorielles : toutes à 0 sauf celle choisie
        edited_client_line = client_line.copy()
        edited_client_line[start + old_value + end] = 0
        edited_client_line[start + value + end] = 1
    # Si on est devant une features binaire
    elif len(unique_val) == 2:
        value = st.selectbox(
            "Valeur à lui donner", unique_val, index=unique_val.index(old_value)
        )
        edited_client_line = client_line.copy()
        edited_client_line[option] = value
    # Si on est duvant un nombre
    elif is_float(old_value):
        value = st.number_input("Valeur à lui donner", value=float(old_value))
        edited_client_line = client_line.copy()
        edited_client_line[option] = value
    # Sinon
    else:
        value = st.text_input("Valeur à lui donner", value=old_value)
        edited_client_line = client_line.copy()
        edited_client_line[option] = value

    # Prédiction du statut du client modifié
    statut = predict_client(edited_client_line.squeeze())

    # Affichage
    st.markdown(
        "# Statut prédit : "
        + (":green[Accepté]" if statut["result"] == 0 else ":red[Refusé]")
    )
    st.text("Ancien score brut : " + str(old_score))
    st.text("Nouveau score brut : " + str(statut["result_proba"]))
    st.caption(
        "(le client est accepté si son score est ≤ à {:.4f}, sinon il est refusé)".format(
            statut["seuil"], statut["seuil"]
        )
    )
