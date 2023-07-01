import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import matplotlib.pyplot as plt



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
    
    # Création de la sidebar
    st.sidebar.title("Infos client")
    
    sk_id = st.sidebar.text_input('ID Client', placeholder="Entrez l'ID client (exemple : 100002)")
    
    sk_id = show_client(infos_client, sk_id, st.sidebar)
    
    st.sidebar.divider()
    
    if sk_id is None:
        st.stop()
    
    
    # Chargement du gros dataset
    dataset = load_dataset(50000)
    # Création du masque utile pour les graphs
    mask_t0 = (dataset['TARGET'] == 0)
    
    # Chargement de la ligne du client
    client_line = get_client_line(dataset, sk_id)
    # Affichage
    st.text(client_line)
    
    
    
    
    # Création du module d'un graphique
    col1, col2 = st.columns([2, 1])

    # Calcul et création du graphique
    to_plot = [dataset.loc[mask_t0, 'AMT_INCOME_TOTAL'], dataset.loc[~mask_t0, 'AMT_INCOME_TOTAL']]

    fig, ax = plt.subplots()
    ax.hist(to_plot, bins=50, label=["Accepté", "Refusé"], density=True, log=True)
    ax.axvline(x=client_line['AMT_INCOME_TOTAL'], color='r', linestyle='--', label='Client')
    ax.legend()
    
    # Ajout du graphique au module
    col1.subheader("Visualisation de AMT_INCOME_TOTAL")
    col1.pyplot(fig)

    col2.write(
    """<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True)
    col2.markdown("### Importance : :green[faible]")
    col2.markdown("### Effet : :red[negatif] :-1:")
    col2.markdown("### Comment améliorer : Augmenter :arrow_up: :point_up_2: :chart_with_upwards_trend:")
    col2.container()
    
    

if __name__ == '__main__':
    main()