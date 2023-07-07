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

# Chargement des variables en fonction de l'environnement
if os.environ.get('ENVIRONMENT') == 'local':
    load_dotenv('config_local.env')
else:
    load_dotenv('config_production.env')

API_DATAFRAME_URL = os.environ.get('API_DATAFRAME_URL')
API_PLOT_URL = os.environ.get('API_PLOT_URL')
API_PREDICT_URL = os.environ.get('API_PREDICT_URL')
CLIENT_INFO_FILE = os.environ.get('CLIENT_INFO_FILE')
DATASET = os.environ.get('DATASET')

@st.cache_data
def load_data_info():
    df = pd.read_pickle(CLIENT_INFO_FILE)
    return df

@st.cache_data
def load_dataset(sample=None):
    df = pd.read_pickle(DATASET)
    if sample is None:
        return df
    else:
        #return df.sample(sample)
        return df.iloc[:sample]


# Je ne peux pas les mettres dans le main sinon
# le dataset n'est accessible qu'en le donnant en
# parametres aux fonctions. Or Streamlit le gère
# mal et met longtemps à relancer get_client_line

#Mise en forme de la page
st.set_page_config(page_title="Rapport d'analyse du statut client", page_icon='./img/logo.png', layout="wide")

# Chargement du gros dataset
dataset = load_dataset()

# Chargement des données de base
infos_client = load_data_info()
    
    
@st.cache_data
def get_client_line(sk_id):
    index = dataset.SK_ID_CURR.eq(sk_id).argmax()
    return dataset.iloc[index]


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

def show_client(df, sk_id, show):
    if sk_id=="":
        return None
    
    if not sk_id.isdigit():
        show.write('ID client doit être un entier')
        return None
    
    sk_id = int(sk_id)
    
    if len(df.loc[df['SK_ID_CURR']==sk_id]) == 0:
        show.write('ID client introuvable')
        return None
    
    client_line = get_client_line(sk_id)
    
    genre = "Inconu"
    if client_line['CODE_GENDER']==0:
        genre = "Homme"
    elif client_line['CODE_GENDER']==1:
        genre = "Femme"
    else:
        genre = "Autre"
    show.write('Genre : ' + genre)
    show.write('Type de prêt : ' + ('Comptant' if client_line['NAME_CONTRACT_TYPE']==0 else 'Renouvelable'))
    
    statut = predict_client(client_line)['result']
    show.markdown("Statut : " + (":green[Accepté]"if client_line['TARGET']==0 else ':red[Refusé]'))
    show.markdown("# Statut predit : " + (":green[Accepté]"if statut==0 else ':red[Refusé]'))
    return sk_id

    
def main():
    #st.sidebar.write(os.environ.get('API_DATAFRAME_URL'))
    #st.sidebar.write(os.environ.get('API_PLOT_URL'))
    #st.sidebar.write(os.environ.get('CLIENT_INFO_FILE'))
    #st.sidebar.write(os.environ.get('DATASET'))
    
    # Création de la sidebar
    col1, col2, col3= st.sidebar.columns([1, 2, 1])
    col1.write("")
    col2.image('./img/logo.png')
    col3.write("")
    
    st.sidebar.title("Infos client")
    
    sk_id = st.sidebar.text_input('ID Client', placeholder="Entrez l'ID client (exemple : 100002)")
    
    sk_id = show_client(infos_client, sk_id, st.sidebar)
    
    st.sidebar.divider()
    
    # Si l'id client a un soucis on stop tout là
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
        #st.caption("Choisissez entre un résumé du choix de l'algorythme, ou une version où vous pouvez modifier les graphiques et tester d'autres paramètres pour les données client")
    
    # Lance la page sélectionnée
    pages[page].run(dataset, client_line, shap_df, shap_img)
    

if __name__ == '__main__':
    main()