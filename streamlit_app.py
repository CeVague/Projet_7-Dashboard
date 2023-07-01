import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import matplotlib.pyplot as plt

import stlib
import importlib

@st.cache_data
def load_data_info():
    df = pd.read_pickle("C:/Users/Administrateur/Documents/OpenClassrooms/GitHub/Projet_7-Dashboard/data/client_info.pkl")
    return df

@st.cache_data
def load_dataset(sample=None):
    df = pd.read_pickle("C:/Users/Administrateur/Documents/OpenClassrooms/GitHub/Projet_7-Dashboard/data/streamlit_dataset.pkl")
    if sample is None:
        return df
    else:
        #return df.sample(sample)
        return df.iloc[:sample]

@st.cache_data
def get_client_line(df, sk_id):
    index = df.SK_ID_CURR.eq(sk_id).argmax()
    return df.iloc[index]

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
    
    client_line = get_client_line(df, sk_id)
    show.write('Genre : ' + ('Homme' if client_line['CODE_GENDER']==0 else 'Femme'))
    show.write('Type de prêt : ' + ('Comptant' if client_line['NAME_CONTRACT_TYPE']==0 else 'Renouvelable'))
    return sk_id

def main():
    #Mise en forme de la page
    st.set_page_config(page_title="Analyse du statut client", layout="wide")
    
    # Chargement des données de base
    infos_client = load_data_info()
    
    # Chargement du gros dataset
    dataset = load_dataset(50000)
    
    # Création de la sidebar
    st.sidebar.title("Infos client")
    
    sk_id = st.sidebar.text_input('ID Client', placeholder="Entrez l'ID client (exemple : 100002)")
    
    sk_id = show_client(infos_client, sk_id, st.sidebar)
    
    st.sidebar.divider()
    
    # Si l'id client a un soucis on stop tout là
    if sk_id is None:
        st.stop()
    
    
    
    # Chargement de la ligne du client
    client_line = get_client_line(dataset, sk_id)
    
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