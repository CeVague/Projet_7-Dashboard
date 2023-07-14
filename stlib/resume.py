description = "Rapport résumé"


def run(dataset, client_line, shap_df, shap_img):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    # Palette de couleur
    COLOR_A = "#008bfb"  # Accepté
    COLOR_R = "#ff0051"  # Refusé
    COLOR_C = "#000000"  # Client

    # Création du masque pour séléctionner les
    # clients acceptés du dataset
    @st.cache_data
    def get_mask():
        return dataset["TARGET"] == 0

    mask_t0 = get_mask()

    # Fonction de génération du graphique selon la colonne et le type de données
    # C'est le graphique dans la partie gauche
    def determine_best_chart(data, mask_t0, client, alt=False, log=[False, False]):
        num_cols = len(data.columns)

        # Cas où le tableau a une seule colonne
        if num_cols == 1:
            column_name = data.columns[0]
            column_data = data[column_name]

            log = log[0]

            client = client[column_name]

            nb_unique = len(column_data.dropna().unique())

            # Vérifier si les données sont numériques
            if np.issubdtype(column_data.dtype, np.number) and nb_unique > 2:
                # Séparer les données en fonction du masque mask_t0
                data_t0 = column_data[mask_t0]
                data_not_t0 = column_data[~mask_t0]

                if alt:
                    # Utiliser des violinplots pour représenter les distributions
                    violinplot = plt.violinplot(
                        [data_not_t0, data_t0],
                        showmedians=True,
                        showextrema=True,
                        vert=False,
                    )

                    # Définir les couleurs de remplissage pour chaque violon
                    colors = [COLOR_R, COLOR_A]
                    for i, patch in enumerate(violinplot["bodies"]):
                        patch.set_color(colors[i])

                    plt.ylabel("Valeurs")
                    plt.xlabel(column_name)
                    plt.yticks([1, 2], ["Refusé", "Accepté"])

                    plt.axvline(x=client, color=COLOR_C, linestyle="--", label="Client")

                    plt.title("Violinplots de {}".format(column_name))
                    plt.legend()
                else:
                    rge = (column_data.min(), column_data.max())

                    # Utiliser un histogramme pour représenter la distribution des valeurs
                    plt.hist(
                        data_t0,
                        bins=50,
                        color=COLOR_A,
                        range=rge,
                        alpha=0.5,
                        label="Accepté",
                        density=True,
                        log=log,
                    )
                    plt.hist(
                        data_not_t0,
                        bins=50,
                        color=COLOR_R,
                        range=rge,
                        alpha=0.5,
                        label="Refusé",
                        density=True,
                        log=log,
                    )

                    plt.axvline(x=client, color=COLOR_C, linestyle="--", label="Client")

                    plt.ylabel("Fréquence d'apparition")
                    plt.xlabel("Valeur ({})".format(column_name))
                    plt.title("Histograme de {}".format(column_name))
                    plt.legend()
            # Sinon, si elles sont catégorielles ou binaires
            else:
                # Transformation des NaN en string
                column_data = column_data.fillna("Pas de valeur")

                # Rend les valeurs binaires plus compréhenssibles
                if nb_unique == 2:
                    column_data = column_data.replace({1: "True", 0: "False"})
                    client = "True" if client == 1 else "False"

                # Récupération des catégories les plus présentes et les autres
                categories = list(column_data.value_counts().head(20).index)
                not_categories = list(set(column_data.unique()) - set(categories))

                # Séparer les données en fonction du masque mask_t0
                data_t0 = column_data[mask_t0]
                data_not_t0 = column_data[~mask_t0]

                # Compter le nombre d'occurrences de chaque catégorie
                counts_t0 = data_t0.value_counts() / len(data_t0)
                counts_not_t0 = data_not_t0.value_counts() / len(data_not_t0)

                # Ajout de la catégorie 'Autre' s'il y en a trop
                if len(not_categories) > 0:
                    counts_t0.loc["Autre"] = counts_t0.loc[not_categories].sum()
                    counts_not_t0.loc["Autre"] = counts_not_t0.loc[not_categories].sum()

                # Si la valeur du client fait partie des masquées
                if client in not_categories:
                    client = "Autre"

                # Créer les positions pour les barres côte à côte
                positions_t0 = np.arange(len(categories))
                positions_not_t0 = (
                    positions_t0 + 0.35
                )  # Décaler les positions pour les barres de mask_not_t0

                # Créer les diagrammes en barres
                plt.bar(
                    positions_t0,
                    counts_t0.reindex(categories, fill_value=0),
                    width=0.35,
                    label="Accepté",
                    color=COLOR_A,
                    log=log,
                )
                plt.bar(
                    positions_not_t0,
                    counts_not_t0.reindex(categories, fill_value=0),
                    width=0.35,
                    label="Refusé",
                    color=COLOR_R,
                    log=log,
                )

                plt.axvline(
                    x=categories.index(client) + 0.17,
                    color=COLOR_C,
                    linestyle="--",
                    label="Client",
                )

                plt.xticks(positions_t0 + 0.17, categories, rotation=45, ha="right")
                plt.xlabel(column_name)
                plt.ylabel("Occurrences (fréquence)")
                plt.title("Diagramme en barres de {}".format(column_name))
                plt.legend()

        # Cas où le tableau a deux colonnes
        elif num_cols == 2:
            # Récupération des noms des colonnes
            x_col = data.columns[0]
            y_col = data.columns[1]

            # Données client
            x_client = client[x_col]
            y_client = client[y_col]

            # On fait en sorte qu'il y ait autant de données de chaque target
            data_resample = data.loc[~mask_t0].copy()
            data_resample["TARGET"] = "Refusé"

            tmp = data.loc[mask_t0]
            tmp = tmp.sample(len(data_resample))
            tmp["TARGET"] = "Accepté"

            data_resample = pd.concat([data_resample, tmp])

            # labels = ['Accepté' if m else 'Refusé' for m in mask_t0]
            labels = data_resample["TARGET"]

            # Graphique combiné
            plot = sns.jointplot(
                data=data_resample,
                x=x_col,
                y=y_col,
                hue=labels,
                palette={"Accepté": COLOR_A + "20", "Refusé": COLOR_R + "40"},
            )

            # Ajout des repères de l'emplacement du client
            plot.ax_joint.axvline(
                x=x_client, color=COLOR_C, linestyle="--", label="Client"
            )
            plot.ax_marg_x.axvline(x=x_client, color=COLOR_C, linestyle="--")
            plot.ax_joint.axhline(y=y_client, color=COLOR_C, linestyle="--")
            plot.ax_marg_y.axhline(y=y_client, color=COLOR_C, linestyle="--")

            plot.ax_joint.plot(x_client, y_client, "o", color=COLOR_C)

            if log[0]:
                plot.ax_joint.set_xscale("log")
            if log[1]:
                plot.ax_joint.set_yscale("log")

            plt.legend()
        else:
            st.write("Erreur lors de la génération du graphique")

    # Récupération de l'influence de cette feature donnée par SHAP
    def get_importance(name_col, val):
        # Récupération si c'est une features étudiée par SHAP
        if name_col in shap_df.index:
            return shap_df.loc[name_col, "shap"]
        # Récupération si c'est une variable catégorielle
        else:
            val = str(val)
            cols = list(shap_df.index)

            # Recupération du nom global de la colonne
            i = name_col.rfind("_")
            start = name_col[: i + 1]
            end = name_col[i:]

            # Liste des toutes les valeurs possibles de cette colonne
            cols = [c for c in cols if c.startswith(start) and c.endswith(end)]

            # Retourne la statistique globale (moyenne de toute les catégories)
            if val != "nan":
                name_col = start + val + end
                cols.remove(name_col)

                return shap_df.loc[name_col, "shap"] - shap_df.loc[cols, "shap"].mean()
            else:
                return shap_df.loc[cols, "shap"].mean()

    # Présentation de la page
    st.title(description)

    st.markdown(
        "Ce rapport résume les raisons de l'acceptation ou du refus du client dont l'identifiant a été saisi."
    )
    st.markdown(
        "Pour faciliter son interprétation, la liste des premières caractéristiques les plus influentes sur ce choix est affichée en premier, ainsi que leur effet."
    )
    st.markdown(
        "Ensuite, les 10 features les plus importantes s'affichent sous forme de graphique, présentant chaque client par rapport au groupe des clients acceptés versus les clients refusés."
    )
    st.markdown(
        "Enfin, à droite de chaque graphique est indiquée l'importance de cette variable, c'est-à-dire son influence sur le résultat final, ainsi que son effet (positif s'il joue en sa faveur, négatif sinon)."
    )

    # Affichage du waterfall de SHAP pour ce client
    st.header("Apercu des features")
    st.image(
        shap_img,
        caption="Liste des features ayant eu le plus d'influence sur le choix final de l'algorythme, ainsi que si l'effet est positif ou négatif",
    )

    # Affichage des différentes visualisations des features par ordre d'importance
    st.header("Visualisation individuelle des features")

    # Descritpion comprehenssible des features
    descriptions = {
        "ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM": "lkjlklklk",
        "AMT_CREDIT__AMT_GOODS_PRICE": "Différence entre le cout du crédit et le prix du produit",
        "AMT_GOODS_PRICE": "Prix du produit acheté",
        "EXT_SOURCE_MEAN": "Moyenne des scores normalisés de 3 sources externes",
        "AMT_CREDIT_r_AMT_INCOME_TOTAL": "Ratio entre le cout du crédit et les revenus du client",
        "AMT_CREDIT": "Cout total du crédit",
        "AMT_INCOME_TOTAL": "Revenus du client",
        "PREV_NAME_SELLER_INDUSTRY_MEDIAN": "Domaine d'industrie le plus fréquent des précédents vendeurs",
        "FLAG_PHONE": "Le client a-t-il un téléphone?",
        "DAYS_BIRTH": "Nombre de jours entre aujourd'hui et la naissance du client",
    }

    # Features ayant besoin d'un passage au log
    feat_log = {"EXT_SOURCE_MEAN": False}

    # Features utilisant un graphique alternatif
    feat_alt = {}

    # Features à combiner
    feat_join = {
        "AMT_CREDIT__AMT_GOODS_PRICE": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
    }

    # On liste les features que l'on souhaite afficher
    # par ordre d'importance
    liste_a_afficher = [
        "AMT_CREDIT_r_AMT_INCOME_TOTAL",
        "EXT_SOURCE_MEAN",
        "FLAG_PHONE",
        "DAYS_BIRTH",
        "AMT_CREDIT",
    ]

    liste_a_afficher = list(shap_df.index)[:10]

    # Pour chaque feature
    for nom_col in liste_a_afficher:

        # On regarde si on doit l'afficher seule ou en 2D
        if nom_col in feat_join:
            feat_1, feat_2 = feat_join[nom_col]
        else:
            feat_1, feat_2 = nom_col, None

        # Création du module en deux partie des visualisations
        col1, col2 = st.columns([2, 1])

        # Colonne de gauche
        # Affichage de la feature traitée
        col1.subheader("Visualisation de " + nom_col)
        if nom_col in descriptions:
            col1.caption(descriptions[nom_col])

        # Affichage du graphique pour une ou deux features
        fig, ax = plt.subplots(figsize=(10, 6))
        if feat_2 is None:
            val_client = client_line[[feat_1]]
            alt = feat_alt[feat_1] if feat_1 in feat_alt else False
            feat_1_log = feat_log[feat_1] if feat_1 in feat_log else True

            determine_best_chart(
                dataset[[feat_1]], mask_t0, val_client, alt, [feat_1_log]
            )
        else:
            val_client = client_line[[feat_1, feat_2]]
            alt = feat_alt[feat_1] if feat_1 in feat_alt else False
            feat_1_log = feat_log[feat_1] if feat_1 in feat_log else True
            feat_2_log = feat_log[feat_1] if feat_2 in feat_log else True

            determine_best_chart(
                dataset[[feat_1, feat_2]],
                mask_t0,
                val_client,
                alt,
                [feat_1_log, feat_2_log],
            )

        col1.pyplot(plt)

        # Colonne de doite
        # Centrage
        col2.write(
            """<style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Récupération de la valeur SHAP liée à cette feature du client
        shap_value = get_importance(nom_col, client_line[nom_col])

        # Affichage de son influence
        if abs(shap_value) > 0.9:
            col2.markdown("### Importance : :red[primordiale]")
        elif abs(shap_value) > 0.15:
            col2.markdown("### Importance : :red[forte]")
        elif abs(shap_value) > 0.07:
            col2.markdown("### Importance : :orange[moyenne]")
        else:
            col2.markdown(
                '### Importance : <span style="color:goldenrod;">faible</span>',
                unsafe_allow_html=True,
            )

        if shap_value > 0:
            col2.markdown("### Effet : :red[négatif] :-1:")
        else:
            col2.markdown("### Effet : :green[positif] :+1:")
