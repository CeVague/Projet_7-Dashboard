import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import matplotlib.pyplot as plt

import pickle

from dotenv import load_dotenv
import os
import io
from PIL import Image

from stlib import edit_client, resume, personalised_graph

import requests

# ------------- Chargement de variables et de données ---------------
    
# Palette de couleur
COLOR_A = '#008bfb' # Accepté
COLOR_R = '#ff0051' # Refusé
COLOR_C = '#000000' # Client

# Chargement des variables en fonction de l'environnement
if os.environ.get('ENVIRONMENT') == 'local':
    load_dotenv('config_local.env')
else:
    load_dotenv('config_production.env')

# Chargement des chemins utiles qui viennent d'être chargés
API_DATAFRAME_URL = os.environ.get('API_DATAFRAME_URL')
API_PLOT_URL = os.environ.get('API_PLOT_URL')
API_PREDICT_URL = os.environ.get('API_PREDICT_URL')
CLIENT_INFO_FILE = os.environ.get('CLIENT_INFO_FILE')
DATASET = os.environ.get('DATASET')

# Chargement des infos clients (les quelques informations
# données pour s'assurer que l'on parle du bon client)
def load_data_info():
    df = pd.read_pickle(CLIENT_INFO_FILE)
    return df

# Chargement du dataset complet ou d'un extrait selon les besoins
def load_dataset(sample=None):
    df = pd.read_pickle(DATASET)
    if sample is None:
        return df
    else:
        # Sample sans mélange pour connaitre les ID clients présentes
        return df.iloc[:sample]


# Je ne peux pas les mettres dans le main sinon
# le dataset n'est accessible qu'en le donnant en
# parametres aux fonctions. Or Streamlit le gère
# mal et met longtemps à relancer get_client_line
# Du coup pas besoin de mise en cache non plus

# Chargement des données de base
infos_client = load_data_info()

# Chargement du gros dataset
dataset = load_dataset()

# ---------- Interaction avec l'API -------------
@st.cache_data
def predict_client(client_line):
    reponse = requests.get(API_PREDICT_URL, json={'data': client_line.to_json(default_handler=str)})
    
    # Vérifier la réponse du serveur
    if reponse.status_code == 200:
        # Récupérer le DataFrame depuis la réponse JSON
        json_reponse = reponse.json()
        
        return json_reponse
    else:
        st.error('Erreur lors de la prédiction')

@st.cache_data
def get_client_shap(client_line):
    reponse = requests.get(API_DATAFRAME_URL, json={'data': client_line.to_json(default_handler=str)})
    
    # Vérifier la réponse du serveur
    if reponse.status_code == 200:
        # Récupérer le DataFrame depuis la réponse JSON
        json_reponse = reponse.json()
        df_reponse = pd.DataFrame.from_dict(json_reponse)

        # Faire quelque chose avec le DataFrame de réponse
        return df_reponse
    else:
        st.error('Erreur lors de la récupération du dataframe')

@st.cache_data
def get_client_shap_plot(client_line, forme='waterfall'):
    reponse = requests.get(API_PLOT_URL+'/'+forme, json={'data': client_line.to_json(default_handler=str)})

    # Vérifier le code de réponse de la requête
    if reponse.status_code == 200:
        image_bytes = reponse.content

        # Affichage de l'image dans Streamlit
        image = Image.open(io.BytesIO(image_bytes))
        return image
    else:
        st.error('Erreur lors de la récupération de l\'image')
        return None


# -------- Fonctions utiles --------------

# Récupère une ligne client d'après son SK_ID (c'est à dire
# toutes les features et leurs valeurs connues d'un client)
@st.cache_data
def get_client_line(sk_id):
    index = dataset.SK_ID_CURR.eq(sk_id).argmax()
    return dataset.iloc[index]

# Affichage des données d'un client
def show_client(sk_id, show):
    # S'il n'y a pas d'ID on renvoie None
    if sk_id=="":
        return None
    
    # Si l'ID n'est pas un chiffre on prévient et renvoie None
    if not sk_id.isdigit():
        show.write('ID client doit être un entier')
        return None
    
    # Sinon c'est un entier donc on le converti
    sk_id = int(sk_id)
    
    # Si aucun client ne correspond on prévient et renvoie None
    if len(infos_client.loc[infos_client['SK_ID_CURR']==sk_id]) == 0:
        show.write('ID client introuvable')
        return None
    
    # Sinon on récupère les infos du client
    client_line = get_client_line(sk_id)
    
    # Et on affiche ses infos
    genre = "Inconu"
    if client_line['CODE_GENDER']==0:
        genre = "Homme"
    elif client_line['CODE_GENDER']==1:
        genre = "Femme"
    else:
        genre = "Autre"
    show.write('Genre : ' + genre)
    show.write('Type de prêt : ' + ('Comptant' if client_line['NAME_CONTRACT_TYPE']==0 else 'Renouvelable'))
    
    # On affiche aussi le statut prédit par le modèle
    statut = predict_client(client_line)['result']
    #show.markdown("Statut : " + (":green[Accepté]"if client_line['TARGET']==0 else ':red[Refusé]'))
    show.markdown("# Statut prédit : " + (":green[Accepté]"if statut==0 else ':red[Refusé]'))
    
    # On renvoie l'ID pour signifier que tout s'est bien passé
    return sk_id

    
def main():
    #Mise en forme de la page
    st.set_page_config(page_title="Rapport d'analyse du statut client", page_icon='./img/logo.png', layout="wide")
    
    # Création de la sidebar
    # Avec le logo
    col1, col2, col3= st.sidebar.columns([1, 6, 1])
    col1.write("")
    col2.image('./img/logo.png')
    col3.write("")
    
    # La texte input et les infos clients récupérées
    st.sidebar.title("Infos client")
    
    sk_id = st.sidebar.text_input('ID Client', placeholder="Entrez l'ID client (exemple : 100002)")
    
    sk_id = show_client(sk_id, st.sidebar)
    
    st.sidebar.divider()
    
    # Si l'id client a un soucis on stop tout ici
    if sk_id is None:
        st.stop()
    
    # Chargement de la ligne du client
    client_line = get_client_line(sk_id)
    
    # Récupération des données Shap
    shap_df = get_client_shap(client_line)
    shap_img = get_client_shap_plot(client_line)
    
    # Chargement des autres pages sous forme de modules
    pages = {'resume':resume, 'edit_client':edit_client, 'personalised_graph':personalised_graph}
    
    # Affiche le menu de sélection de page
    with st.sidebar:
        page = st.selectbox("Mode d'analyse:", pages.keys(), format_func=lambda k:pages[k].description)
    
    # Lance la page sélectionnée en lui passant les données chargées
    pages[page].run(dataset, client_line, shap_df, shap_img)
    

if __name__ == '__main__':
    main()