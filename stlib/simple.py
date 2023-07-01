description = "Rapport simple"

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
    st.title("Rapport simple")
    
    
    
    
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


# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()