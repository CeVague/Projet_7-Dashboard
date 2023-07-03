description = "Rapport simple"

def run(dataset, client_line):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    @st.cache_data
    def get_mask():
        return dataset['TARGET'] == 0
    
    mask_t0 = get_mask()
    
    @st.cache_resource
    def get_fig(name_col, dtype=None):
        fig, ax = plt.subplots()
        to_plot = [dataset.loc[mask_t0, name_col].dropna(), dataset.loc[~mask_t0, name_col].dropna()]
        if name_col == 'AMT_INCOME_TOTAL':
            ax.hist(to_plot, bins=50, label=["Accepté", "Refusé"], density=True, log=True)
        elif dtype == 'cat_0':
            ax.hist(to_plot, bins=len(to_plot[0].unique())-1, label=["Accepté", "Refusé"], density=True, log=True, align='left')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        elif dtype == 'num_0':
            ax.boxplot(to_plot, vert=False, labels=['Acceptés', 'Refusés'])
        else:
            # Violinplot pour la première distribution
            violin0 = ax.violinplot(to_plot[0], positions=[1], widths=0.5, vert=False, showmeans=False, showmedians=True)
            violin0['bodies'][0].set_facecolor('skyblue')

            # Violinplot pour la deuxième distribution
            violin1 = ax.violinplot(to_plot[1], positions=[2], widths=0.5, vert=False, showmeans=False, showmedians=True)
            violin1['bodies'][0].set_facecolor('lightgreen')

            
            # Configurations des axes
            ax.set_yticks([1, 2])
            ax.set_yticklabels(["Acceptés", "Refusés"])
            ax.set_xlabel('Valeurs')
        
        ax.axvline(x=client_line[name_col], color='r', linestyle='--', label='Client')
        ax.legend()
        
        return fig, ax
    
    # Affichage
    st.title("Rapport simple")
    
    traduction = {
        'EXT_SOURCE_MEAN': 'Moyenne des scores normalisés de 3 sources externes',
        'AMT_INCOME_TOTAL': 'Revenus du client',
        'NAME_EDUCATION_TYPE': 'Plus haut niveau de diplome du client',
    }
    
    
    
    for name_col, dtype in [('AMT_INCOME_TOTAL', None),
                            ('EXT_SOURCE_MEAN', 'num_0'),
                            ('EXT_SOURCE_MEAN', None),
                            ('NAME_EDUCATION_TYPE', 'cat_0')
                           ]:
    
        # Création du module d'un graphique
        col1, col2 = st.columns([2, 1])

        fig, ax = get_fig(name_col, dtype)

        # Ajout du graphique au module
        col1.subheader("Visualisation de "+name_col)
        col1.caption(traduction[name_col])
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