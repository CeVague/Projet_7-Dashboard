description = "Graphiques personalisés"

def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    
    # Palette de couleur
    COLOR_A = '#008bfb' # Accepté
    COLOR_R = '#ff0051' # Refusé
    COLOR_C = '#000000' # Client
    
    @st.cache_data
    def get_mask():
        return dataset['TARGET'] == 0
    
    mask_t0 = get_mask()

    def determine_best_chart(data, mask_t0, alt=False):
        num_cols = len(data.columns)

        # Cas où le tableau a une seule colonne
        if num_cols == 1:
            column_name = data.columns[0]
            column_data = data[column_name]

            # Vérifier si les données sont numériques
            if np.issubdtype(column_data.dtype, np.number) and len(column_data.unique())>2:
                # Séparer les données en fonction du masque mask_t0
                data_t0 = column_data[mask_t0]
                data_not_t0 = column_data[~mask_t0]
                
                
                if alt:
                    # Utiliser des violinplots pour représenter les distributions
                    violinplot = plt.violinplot([data_not_t0, data_t0], showmedians=True, showextrema=True, vert=False)

                    # Définir les couleurs de remplissage pour chaque violon
                    colors = [COLOR_R, COLOR_A]
                    for i, patch in enumerate(violinplot['bodies']):
                        patch.set_color(colors[i])

                    plt.ylabel('Valeurs')
                    plt.xlabel(column_name)
                    plt.yticks([1, 2], ['Refusé', 'Accepté'])
                    plt.title('Violinplots de {}'.format(column_name))
                else:
                    rge = (column_data.min(), column_data.max())

                    # Utiliser un histogramme pour représenter la distribution des valeurs
                    plt.hist(data_t0, bins=20, color=COLOR_A, range=rge, alpha=0.5, label='Accepté', density=True)
                    plt.hist(data_not_t0, bins=20, color=COLOR_R, range=rge, alpha=0.5, label='Refusé', density=True)

                    plt.ylabel('Fréquence d\'apparition')
                    plt.xlabel('Valeur ({})'.format(column_name))
                    plt.title('Histograme de {}'.format(column_name))
                    plt.legend()
            else:
                # Transformation des NaN en string
                column_data = column_data.fillna('Pas de valeur')
                
                # Récupération des catégories les plus présentes et les autres
                categories = list(column_data.value_counts().head(20).index)
                not_categories = list(set(column_data.unique()) - set(categories))
                
                # Séparer les données en fonction du masque mask_t0
                data_t0 = column_data[mask_t0]
                data_not_t0 = column_data[~mask_t0]

                # Compter le nombre d'occurrences de chaque catégorie
                counts_t0 = data_t0.value_counts()/len(data_t0)
                counts_not_t0 = data_not_t0.value_counts()/len(data_not_t0)
                
                # Ajout de la catégorie 'Autre' s'il y en a trop
                if len(not_categories) > 0:
                    counts_t0.loc['Autre'] = counts_t0.loc[not_categories].sum()
                    counts_not_t0.loc['Autre'] = counts_not_t0.loc[not_categories].sum()

                # Créer les positions pour les barres côte à côte
                positions_t0 = np.arange(len(categories))
                positions_not_t0 = positions_t0 + 0.35  # Décaler les positions pour les barres de mask_not_t0

                # Créer les diagrammes en barres
                plt.bar(positions_t0, counts_t0.reindex(categories, fill_value=0), width=0.35, label='Accepté', color=COLOR_A)
                plt.bar(positions_not_t0, counts_not_t0.reindex(categories, fill_value=0), width=0.35, label='Refusé', color=COLOR_R)

                plt.xticks(positions_t0 + 0.17, categories, rotation=45, ha='right')
                plt.xlabel(column_name)
                plt.ylabel('Occurrences (fréquence)')
                plt.title('Diagramme en barres de {}'.format(column_name))
                plt.legend()

        # Cas où le tableau a deux colonnes
        elif num_cols == 2:
            x_col = data.columns[0]
            y_col = data.columns[1]
            
            data_resample = data.loc[~mask_t0]
            data_resample = pd.concat([data_resample, data.loc[mask_t0].sample(len(data_resample))])
            
            #labels = ['Accepté' if m else 'Refusé' for m in mask_t0]
            labels = ['Refusé']*(len(data_resample)//2) + ['Accepté']*(len(data_resample)//2)
            
            sns.jointplot(data=data_resample, x=x_col, y=y_col, hue=labels, palette={'Accepté':COLOR_A+'20', 'Refusé':COLOR_R+'40'})
            
            plt.title('Graphique combiné entre {} et {}'.format(x_col, y_col))

        else:
            st.write("Le tableau ne doit avoir qu'une ou deux colonnes.")
    
    list_cols = list(dataset.columns)
    
    # Affichage
    st.title(description)
    
    st.markdown("Cette page offre la possibilité de visualiser les caractéristiques de votre choix sous forme de graphique.")
    
    with st.expander("Apercu des features"):
        st.image(shap_img, caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif")
    
    st.header("Visualisation individuelle des features")

    form = st.form("select_columns", clear_on_submit=False)
    with form:
        col1, col2 = st.columns(2)
        with col1:
            feat_1 = st.selectbox("Première colonne à visualiser", list_cols)
            #feat_1_log = st.checkbox("Axe 1 en log ?")
        with col2:
            feat_2 = st.selectbox("Deuxième colonne à visualiser", ['-----'] + list_cols)
            #feat_2_log = st.checkbox("Axe 2 en log ?")
        alt = st.checkbox("Diagramme alternatif ?")

        # Now add a submit button to the form:
        st.form_submit_button("Actualiser")

    # Affichage du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    if feat_2 not in list_cols:
        determine_best_chart(dataset[[feat_1]], mask_t0, alt)
    else:
        determine_best_chart(dataset[[feat_1, feat_2]], mask_t0, alt)
        
    st.pyplot(plt)
    