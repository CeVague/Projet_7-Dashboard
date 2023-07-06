import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import matplotlib.pyplot as plt

import stlib
import importlib

import pickle

from dotenv import load_dotenv
import os

# Charger les variables d'environnement en fonction de l'environnement
if os.environ.get('ENVIRONMENT') == 'local':
    load_dotenv('config_local.env')
else:
    load_dotenv('config_production.env')

API_URL = os.environ.get('API_URL')
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
st.set_page_config(page_title="Analyse du statut client", layout="wide")

# Chargement du gros dataset
dataset = load_dataset()

# Chargement des données de base
infos_client = load_data_info()
    
    
@st.cache_data
def get_client_line(sk_id):
    index = dataset.SK_ID_CURR.eq(sk_id).argmax()
    return dataset.iloc[index]

@st.cache_data
def get_client_shap(client_line):
    with open('./data/explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('./data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('./data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./data/columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    
    client_line_scaled = scaler.transform([client_line[list(columns)]])
    client_line_explained = explainer(client_line_scaled)
    
    tmp = pd.DataFrame(client_line_explained.values, columns=columns, index=['shap']).T
    tmp['abs'] = tmp['shap'].abs()
    tmp = tmp.sort_values('abs', ascending=False)
    
    return tmp

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
    show.write('Genre : ' + ('Homme' if client_line['CODE_GENDER']==0 else 'Femme'))
    show.write('Type de prêt : ' + ('Comptant' if client_line['NAME_CONTRACT_TYPE']==0 else 'Renouvelable'))
    show.markdown("Statut : " + (":green[Accepté]"if client_line['TARGET']==0 else ':red[Refusé]'))
    return sk_id

    
def main():
    st.sidebar.write(os.environ.get('API_URL'))
    st.sidebar.write(os.environ.get('CLIENT_INFO_FILE'))
    st.sidebar.write(os.environ.get('DATASET'))
    
    # Création de la sidebar
    st.sidebar.title("Infos client")
    
    sk_id = st.sidebar.text_input('ID Client', placeholder="Entrez l'ID client (exemple : 100002)")
    
    sk_id = show_client(infos_client, sk_id, st.sidebar)
    
    st.sidebar.divider()
    
    # Si l'id client a un soucis on stop tout là
    if sk_id is None:
        st.stop()
    
    
    
    # Chargement de la ligne du client
    client_line = get_client_line(sk_id)
    
    
    
    # -------------tests--------------------
    shap_tmp = get_client_shap(client_line)
    st.write(shap_tmp)
    
    
    
    
    
    # Chargement des autres pages sous forme de modules
    moduleNames = ['simple','complexe']
    pages = {}

    # Charge chaque librairie
    for modname in moduleNames:
        m = importlib.import_module('.'+modname,'stlib')
        pages[modname] = m
    
    # Affiche le menu de sélection de page
    with st.sidebar:
        page = st.selectbox("Mode d'analyse:", pages.keys(), format_func=lambda k:pages[k].description) 
    
    # Lance la page sélectionnée
    pages[page].run(dataset, client_line)
    

if __name__ == '__main__':
    main()