description = "Rapport résumé"

def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # Palette de couleur
    COLOR_A = '#008bfb' # Accepté
    COLOR_R = '#ff0051' # Refusé
    COLOR_C = '#000000' # Client
    
    # Création du masque pour séléctionner les 
    # clients acceptés du dataset
    @st.cache_data
    def get_mask():
        return dataset['TARGET'] == 0
    
    mask_t0 = get_mask()
    
    # Fonction de génération du graphique selon la colonne et le type de données
    # C'est le graphique dans la partie gauche
    def get_fig(name_col, dtype=None):
        fig, ax = plt.subplots()
        to_plot = [dataset.loc[mask_t0, name_col].dropna(), dataset.loc[~mask_t0, name_col].dropna()]
        skip_legend=False
        if name_col == 'ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM' or name_col == 'AMT_CREDIT__AMT_GOODS_PRICE' :
            if name_col == 'ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM':
                i = name_col.rfind('_')
                start = name_col[:i+1]
                end = name_col[i:]

                col1 = name_col[:len(start)]+"SUM"
                col2 = name_col[:len(start)]+"MEAN"
            else:
                col1 = 'AMT_CREDIT'
                col2 = 'AMT_GOODS_PRICE'
                
            
            ax.scatter(dataset.loc[mask_t0, col1], dataset.loc[mask_t0, col2], s=1, c='g', label="Accepté") 
            ax.scatter(dataset.loc[~mask_t0, col1], dataset.loc[~mask_t0, col2], s=1, c='r', label="Refusé")
            
            ax.scatter(x=client_line[col1], y=client_line[col2], color='black', label='Client')
            
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            
            ax.legend()
            
            skip_legend=True
        elif name_col == 'AMT_CREDIT_r_AMT_INCOME_TOTAL':
            ax.hist(to_plot, bins=50, range=(0, 20), label=["Accepté", "Refusé"], color=[COLOR_A, COLOR_R], density=True, log=False, histtype='bar', stacked=True, fill=True)
        elif dtype == 'cat_0':
            ax.hist(to_plot, bins=len(to_plot[0].unique())-1, label=["Accepté", "Refusé"], color=[COLOR_A, COLOR_R], density=True, log=True, align='left')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
        elif dtype == 'cat_1':
            
            
            species = ('Non', 'Oui')
            sex_counts = {
                'Accepté': ([to_plot[0].sum()/len(to_plot[0]), 1 - (to_plot[0].sum()/len(to_plot[0]))], COLOR_A),
                'Refusé': ([to_plot[1].sum()/len(to_plot[1]), 1 - (to_plot[1].sum()/len(to_plot[1]))], COLOR_R),
            }
            width = 0.6  # the width of the bars: can also be len(x) sequence

            bottom = np.zeros(2)

            for sex, sex_count in sex_counts.items():
                sex_count, color = sex_count
                p = ax.bar(species, sex_count, width, label=sex, bottom=bottom, color=color)
                bottom += sex_count

                ax.bar_label(p, label_type='center')
        elif dtype == 'num_0':
            ax.boxplot(to_plot, vert=False, labels=['Acceptés', 'Refusés'])
        else:
            # Violinplot pour la première distribution
            violin0 = ax.violinplot(to_plot[0], positions=[1], widths=0.5, vert=False, showmeans=False, showmedians=True)
            violin0['bodies'][0].set_facecolor(COLOR_A)

            # Violinplot pour la deuxième distribution
            violin1 = ax.violinplot(to_plot[1], positions=[2], widths=0.5, vert=False, showmeans=False, showmedians=True)
            violin1['bodies'][0].set_facecolor(COLOR_R)

            
            # Configurations des axes
            ax.set_yticks([1, 2])
            ax.set_yticklabels(["Acceptés", "Refusés"])
            ax.set_xlabel('Valeurs')
        
        if not skip_legend:
            ax.axvline(x=client_line[name_col], color='black', linestyle='--', label='Client')
            ax.legend()
        
        return fig, ax
    
    # Récupération de l'influence de cette feature donnée par SHAP
    def get_importance(name_col, val):
        # Récupération si c'est une features étudiée par SHAP
        if name_col in shap_df.index:
            return shap_df.loc[name_col, 'shap']
        # Récupération si c'est une variable catégorielle
        else:
            val = str(val)
            cols = list(shap_df.index)
            
            # Récupération d
            i = name_col.rfind('_')
            start = name_col[:i+1]
            end = name_col[i:]
            
            cols = [c for c in cols if c.startswith(start) and c.endswith(end)]
            
            if val != 'nan':
                name_col = start+val+end
                cols.remove(name_col)
            
                return shap_df.loc[name_col, 'shap'] - shap_df.loc[cols, 'shap'].mean()
            else:
                return shap_df.loc[cols, 'shap'].mean()
            
    
    # Présentation de la page
    st.title(description)
    
    st.markdown("Ce rapport résume les raisons de l'acceptation ou du refus du client dont l'identifiant a été saisi.")
    st.markdown("Pour faciliter son interprétation, la liste des premières caractéristiques les plus influentes sur ce choix est affichée en premier, ainsi que leur effet.")
    st.markdown("Ensuite, les différents graphiques s'affichent par ordre d'importance, présentant chaque client par rapport au groupe des clients acceptés versus les clients refusés.")
    st.markdown("Enfin, à droite de chaque graphique est indiquée l'importance de cette variable, c'est-à-dire son influence sur le résultat final, ainsi que son effet (positif s'il joue en sa faceur, négatif sinon). Une indication sur une éventuelle évolution de cette valeur est également fournie, suggérant que si cette évolution se produit, le client a plus de chance d'être accepté s'il ne l'est pas déjà.")
    
    # Affichage du waterfall de SHAP pour ce client
    st.header("Apercu des features")
    st.image(shap_img, caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif")
    
    # Affichage des différentes visualisations des features par ordre d'importance
    st.header("Visualisation individuelle des features")
    
    # Descritpion comprehenssible des features
    descriptions = {
        'ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM':"lkjlklklk",
        'AMT_CREDIT__AMT_GOODS_PRICE':"lkjlklklk",
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
                            ('ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM', None),
                            ('AMT_CREDIT__AMT_GOODS_PRICE', None),
                           ]:
    
        # Création du module en deux partie des visualisations
        col1, col2 = st.columns([2, 1])

        # Colonne de gauche
        # Affichage de la feature traitée
        col1.subheader("Visualisation de "+name_col)
        col1.caption(descriptions[name_col])
        
        # Génération du graphique
        fig, ax = get_fig(name_col, dtype)

        # Ajout du graphique au module
        col1.pyplot(fig)

        # Colonne de doite
        # Centrage
        col2.write(
        """<style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True)
        
        # Récupération de la valeur SHAP liée à cette feature du client
        shap_value = get_importance(name_col, client_line[name_col])
        
        # Affichage de son influence
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
        
        #col2.markdown("### Comment améliorer : Augmenter :arrow_up: :point_up_2: :chart_with_upwards_trend:")
        