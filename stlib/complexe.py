description = "Rapport avancé"

def run(dataset, client_line):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt

    @st.cache_data
    def load_dataset(sample=None):
        df = pd.read_pickle("C:/Users/Administrateur/Documents/OpenClassrooms/GitHub/Projet_7-Dashboard/data/streamlit_dataset.pkl")
        if sample is None:
            return df
        else:
            #return df.sample(sample)
            return df.iloc[:sample]

    
    mask_t0 = (dataset['TARGET'] == 0)
    
    # Affichage
    st.title("Rapport B")
    
# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()