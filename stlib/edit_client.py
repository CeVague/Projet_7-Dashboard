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
    
    st.markdown("Ce rapport permet deux choses :")
    st.markdown("Soit de regarder d'autres features du client avec des graphiques personalisés selon vos choix")
    st.markdown("Soit de modifier certaines infos du client et observer son effet sur les prédiction")
    
    with st.expander("Apercu des features"):
        st.write(shap_df)
        st.image(shap_img, caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif")
    
    
    
    options = list(client_line.index)
    options = [c for c in options if c not in list(resume_cat_col['complet'])]

    old_score = predict_client(client_line.squeeze())['result_proba']
    option = st.selectbox("Feature à modifier", client_line.index)
    old_value = client_line[option]
    st.text('Valeur actuelle : '+str(old_value))
    if option in list(resume_cat_col['start']):
        value = st.selectbox('Valeur à lui donner', resume_cat_col.loc[resume_cat_col['start']==option, 'val'])
        edited_client_line = client_line.copy()
        edited_client_line[option+'_'+old_value] = 0
        edited_client_line[option+'_'+value] = 1
    elif is_float(old_value):
        value = st.number_input('Valeur à lui donner')
        edited_client_line = client_line.copy()
        edited_client_line[option] = value
    else:
        value = st.text_input('Valeur à lui donner')
        edited_client_line = client_line.copy()
        edited_client_line[option] = value

    statut = predict_client(edited_client_line.squeeze())
    st.markdown("# Statut predit : " + (":green[Accepté]"if statut['result']==0 else ':red[Refusé]'))
    st.text("Ancien score brut : " + str(old_score))
    st.text("Nouveau score brut : " + str(statut['result_proba']))

    
# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()