description = "Rapport simple"

def run(dataset, client_line, shap_tmp):
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
        if name_col == 'AMT_CREDIT_r_AMT_INCOME_TOTAL':
            ax.hist(to_plot, bins=50, label=["Accepté", "Refusé"], density=True, log=True)
        elif dtype == 'cat_0':
            ax.hist(to_plot, bins=len(to_plot[0].unique())-1, label=["Accepté", "Refusé"], density=True, log=True, align='left')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        elif dtype == 'cat_1':
            
            
            species = ('Non', 'Oui')
            sex_counts = {
                'Accepté': [to_plot[0].sum()/len(to_plot[0]), 1 - (to_plot[0].sum()/len(to_plot[0]))],
                'Refusé': [to_plot[1].sum()/len(to_plot[1]), 1 - (to_plot[1].sum()/len(to_plot[1]))],
            }
            width = 0.6  # the width of the bars: can also be len(x) sequence

            bottom = np.zeros(2)

            for sex, sex_count in sex_counts.items():
                p = ax.bar(species, sex_count, width, label=sex, bottom=bottom)
                bottom += sex_count

                ax.bar_label(p, label_type='center')
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
    
    @st.cache_data
    def get_importance(name_col, val):
        if name_col in shap_tmp.index:
            return shap_tmp.loc[name_col, 'shap']
        else:
            cols = list(shap_tmp.index)
            i = name_col.rfind('_')
            start = name_col[:i+1]
            end = name_col[i:]
            
            name_col = start+val+end
            
            cols = [c for c in cols if c.startswith(start) and c.endswith(end)]
            cols.remove(name_col)
            
            return shap_tmp.loc[name_col, 'shap'] - shap_tmp.loc[cols, 'shap'].mean()
            
    
    # Affichage
    st.title("Rapport simple")
    
    traduction = {
        'EXT_SOURCE_MEAN': 'Moyenne des scores normalisés de 3 sources externes',
        'AMT_CREDIT_r_AMT_INCOME_TOTAL': 'Ratio entre le cout du crédit et les revenus du client',
        'PREV_NAME_SELLER_INDUSTRY_MEDIAN': 'Domaine d\'industrie le plus fréquent des précédents vendeurs',
        'FLAG_PHONE': 'Le client as-t-il un téléphone?',
        'DAYS_BIRTH': 'Age (en jours)',
    }
    
    
    
    for name_col, dtype in [('AMT_CREDIT_r_AMT_INCOME_TOTAL', None),
                            ('EXT_SOURCE_MEAN', 'num_0'),
                            ('EXT_SOURCE_MEAN', None),
                            ('PREV_NAME_SELLER_INDUSTRY_MEDIAN', 'cat_0'),
                            ('FLAG_PHONE', 'cat_1'),
                            ('DAYS_BIRTH', None),
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
        
        shap_value = get_importance(name_col, client_line[name_col])
        col2.write(shap_value)
        
        if abs(shap_value)>0.15:
            col2.markdown("### Importance : :red[forte]")
        elif abs(shap_value)>0.05:
            col2.markdown("### Importance : :orange[moyenne]")
        else:
            col2.markdown("### Importance : :green[faible]")
        
        if shap_value>0:
            col2.markdown("### Effet : :red[negatif] :-1:")
        else:
            col2.markdown("### Effet : :green[positif] :+1:")
        
        col2.markdown("### Comment améliorer : Augmenter :arrow_up: :point_up_2: :chart_with_upwards_trend:")
        col2.container()


# This code allows you to run the app standalone
# as well as part of a library of apps
if __name__ == "__main__":
    run()