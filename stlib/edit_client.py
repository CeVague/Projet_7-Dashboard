description = "Modification client"

def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import os
    
    import requests
    
    
    API_PREDICT_URL = os.environ.get('API_PREDICT_URL')
    RESUME_CAT_COL = os.environ.get('RESUME_CAT_COL')
    
    
    # Recupère la liste des colonnes catégorielles
    # et les versions OneHotEncodées reliées
    @st.cache_data
    def load_resume_cat_col():
        df = pd.read_pickle(RESUME_CAT_COL)
        return df
    
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
            
    def is_float(v):
        try:
            f=float(v)
        except ValueError:
            return False
        return True
    
    resume_cat_col = load_resume_cat_col()
    
    # Affichage
    st.title(description)
    
    st.markdown("Cette page offre la possibilité de modifier les caractéristiques sur lesquelles le client a une influence afin de réaliser une simulation et de voir s'il pourrait être accepté s'il ne l'est pas déjà, ou s'il pourrait modifier certains termes de son contrat tout en restant éligible.")
    st.markdown("Il est important de noter que cette prédiction est une simulation et ne prend pas en compte les relations entre les différentes caractéristiques. Une véritable modification et simulation sont nécessaires pour valider définitivement ces changements.")
    
    with st.expander("Apercu des features"):
        #st.write(shap_df)
        st.image(shap_img, caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif")
    
    
    # On récupère la liste des features
    options = list(client_line.index)
    # Et on retire toutes les OneHotEncoded pour ajouter les catégorielles
    options = [c for c in options if not c.startswith(tuple(resume_cat_col['start']))]
    options += list(resume_cat_col['simple'].unique())

    old_score = predict_client(client_line.squeeze())['result_proba']
    option = st.selectbox("Feature à modifier", options)
    old_value = client_line[option]
    st.text('Valeur d\'origine : '+str(old_value))
    if option in list(resume_cat_col['simple']):
        ligne_option = resume_cat_col.loc[resume_cat_col['simple'] == option].iloc[0]
        start = ligne_option['start'] + '_'
        end = ligne_option['end']
        
        list_option = list(resume_cat_col.loc[resume_cat_col['simple']==option, 'val'])
        
        value = st.selectbox('Valeur à lui donner', list_option, index=list_option.index(old_value))
        edited_client_line = client_line.copy()
        edited_client_line[start+old_value+end] = 0
        edited_client_line[start+value+end] = 1
    elif is_float(old_value):
        value = st.number_input('Valeur à lui donner', value=float(old_value))
        edited_client_line = client_line.copy()
        edited_client_line[option] = value
    else:
        value = st.text_input('Valeur à lui donner', value=old_value)
        edited_client_line = client_line.copy()
        edited_client_line[option] = value

    statut = predict_client(edited_client_line.squeeze())
    st.markdown("# Statut prédit : " + (":green[Accepté]"if statut['result']==0 else ':red[Refusé]'))
    st.text("Ancien score brut : " + str(old_score))
    st.text("Nouveau score brut : " + str(statut['result_proba']))
