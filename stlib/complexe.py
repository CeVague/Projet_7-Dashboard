description = "Rapport avancé"

def run(dataset, client_line):
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
    st.title("Rapport B")
    
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
        
    
# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()