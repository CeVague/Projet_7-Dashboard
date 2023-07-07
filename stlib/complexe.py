description = "Rapport interactif"

def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import os
    
    import requests
    
    
    API_PREDICT_URL = os.environ.get('API_PREDICT_URL')
    
    @st.cache_data
    def get_mask():
        return dataset['TARGET'] == 0
    
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
    
    mask_t0 = get_mask()
    
    list_cols = list(dataset.columns)
    
    # Affichage
    st.title(description)
    
    st.markdown("Ce rapport permet deux choses :")
    st.markdown("Soit de regarder d'autres features du client avec des graphiques personalisés selon vos choix")
    st.markdown("Soit de modifier certaines infos du client et observer son effet sur les prédiction")
    
    with st.expander("Apercu des features"):
        st.write(shap_df)
        st.image(shap_img, caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif")
    
    
    tab1, tab2 = st.tabs(["Modification", "Visualisation"])
    
    with tab2:
        st.header("Visualisation individuelle des features")

        form = st.form("select_columns", clear_on_submit=False)
        with form:
            col1, col2 = st.columns(2)
            with col1:
                feat_1 = st.selectbox("Première colonne à visualiser", list_cols)
            with col2:
                feat_2 = st.selectbox("Deuxième colonne à visualiser", ['-----'] + list_cols)
            inverse = st.checkbox("Inverser les axes ?")

            # Now add a submit button to the form:
            st.form_submit_button("Actualiser")

        # Affichage du graphique
        fig, ax = plt.subplots()
        if feat_2 not in list_cols:
            dataset[feat_1].dropna().plot()
        else:
            dataset[[feat_1, feat_2]].dropna().plot()
        st.pyplot()
        
    with tab1:
        old_score = predict_client(client_line.squeeze())['result_proba']
        option = st.selectbox("Feature à modifier", client_line.index)
        old_value = client_line[option]
        st.text('Valeur actuelle : '+str(old_value))
        if type(old_value) == type(0.01):
            value = st.number_input('Valeur à lui donner')
        else:
            value = st.text_input('Valeur à lui donner')
        #edited_client_line = st.data_editor(client_line.to_frame())
        
        if st.button('Modifier'):
            edited_client_line = client_line.copy()
            edited_client_line[option] = eval(value)
        else:
            edited_client_line = client_line.copy()
        
        statut = predict_client(edited_client_line.squeeze())
        st.markdown("# Statut predit : " + (":green[Accepté]"if statut['result']==0 else ':red[Refusé]'))
        st.text("Ancien score brut : " + str(old_score))
        st.text("Nouveau score brut : " + str(statut['result_proba']))

        #favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
        #st.markdown(f"Your favorite command is **{favorite_command}** 🎈")
    
# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()