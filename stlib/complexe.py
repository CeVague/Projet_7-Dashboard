description = "Rapport interactif"

def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    
    @st.cache_data
    def get_mask():
        return dataset['TARGET'] == 0
    
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
    
    
    tab1, tab2 = st.tabs(["Visualisation", "Modification"])
    
    with tab1:
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
        
    with tab2:
        print(client_line)
        edited_df = st.data_editor(client_line)

        #favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
        #st.markdown(f"Your favorite command is **{favorite_command}** 🎈")
    
# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()