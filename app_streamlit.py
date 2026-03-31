import streamlit as st
from PIL import Image
import io
import os
import json
import numpy as np

from recherche_image import calculer_map, calculer_map_toutes_combinaisons, afficher_images_proches

from descripteurs import (
    calculer_histogramme_couleur,
    calculer_histogramme_gris,
    calculer_histogramme_hsv,
    calculer_histogramme_indexe,
    calculer_histogramme_cumule_gris,
    calculer_histogramme_cumule_couleur,
    calculer_histogramme_cumule_indexe,
    calculer_lbp_histogramme,
    calculer_entropie,
    calculer_image_indexee_descripteur,
    calculer_histogramme_bloc,
    calculer_csv,
    calculer_dcd,
    calculer_ccd,
    calculer_matrice_concurrence,
    calculer_lbp,
    calculer_histogramme_bloc_lbp,
    calculer_histogramme_direction_gradient,
    calculer_histogramme_pondere_par_norme,
    calculer_histogramme_bloc_direction_gradient,
)
from distances import (
    distance_l1,
    distance_l2,
    distance_cosinus,
    distance_chi2,
    distance_intersection,
)
from espaces_couleurs import (
    rgb_vers_gris_moyenne,
    rgb_vers_gris_bt601,
    rgb_vers_gris_bt709,
    rgb_vers_hsv,
    rgb_vers_hls,
    rgb_vers_ycrcb,
    rgb_vers_lab,
    rgb_vers_luv,
    rgb_vers_xyz,
    rgb_vers_bgr,
    rgb_vers_image_indexee,
)
from traitement_image import (
    filtre_prewitt,
    filtre_sobel,
    filtre_roberts,
    filtre_laplacien,
    filtre_scharr,
    creation_filtre_aleatoire,
    cross_correlation,
    convolution,
    ajouter_bruit_gaussien,
    ajouter_bruit_poivre_sel,
    ajouter_bruit_poisson,
    ajouter_padding,
)
from segmentation import (
    binarisation_niveaux_gris,
    binarisation_gris_moyenne,
    binarisation_gris_mediane,
    binarisation_gris_minmax,
    binarisation_gris_ecart_type,
    binarisation_gris_ptile,
    binarisation_gris_moyenne_tronquee,
    binarisation_teinte,
    binarisation_hue_moyenne,
    binarisation_hue_mediane,
    binarisation_hue_minmax,
    binarisation_hue_ecart_type,
    binarisation_hue_ptile,
    binarisation_hue_moyenne_tronquee,
    binarisation_isodata_moyenne,
    binarisation_isodata_4coins,
    binarisation_locale_moyenne,
    binarisation_locale_mediane,
    binarisation_locale_minmax,
    binarisation_adaptatif,
    up_sampling,
    segmentation_kmeans,
    segmentation_fcn,
    segmentation_unet,
    segmentation_segnet,
    segmentation_pspnet,
    calculer_statistiques_binarisation,
)

# Configuration de la page
st.set_page_config(page_title="Projet Vision - Interface", layout="wide")

# Barre latérale de navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    (
        "Accueil",
        "Traitement d'image",
        "Moteur de recherche",
        "Binarisation"
    )
)

# ── Sélecteur d'image via la base ──────────────────────────────────────────
DOSSIER_BASE = "BD_images_resized"

dossiers = sorted([
    d for d in os.listdir(DOSSIER_BASE)
    if os.path.isdir(os.path.join(DOSSIER_BASE, d))
]) if os.path.exists(DOSSIER_BASE) else []

st.sidebar.markdown("---")
st.sidebar.subheader("Sélection d'image")

dossier_choisi = st.sidebar.selectbox("Catégorie", dossiers)

images_disponibles = []
if dossier_choisi:
    dossier_path = os.path.join(DOSSIER_BASE, dossier_choisi)
    images_disponibles = sorted([
        f for f in os.listdir(dossier_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

image_choisie = st.sidebar.selectbox("Image", images_disponibles)

image_path = os.path.join(DOSSIER_BASE, dossier_choisi, image_choisie) if dossier_choisi and image_choisie else None

if image_path:
    st.sidebar.markdown(f"**Chemin :** `{image_path}`")


def charger_image_traitement(image_path_local, fichier_uploade):
    if fichier_uploade is not None:
        return Image.open(fichier_uploade).convert("RGB")
    if image_path_local:
        return Image.open(image_path_local).convert("RGB")
    return None


def normaliser_pour_affichage(image_array):
    img = image_array.astype(np.float32)
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - min_val) / (max_val - min_val)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def image_array_vers_bytes_png(image_array):
    image_uint8 = image_array
    if image_uint8.dtype != np.uint8:
        image_uint8 = normaliser_pour_affichage(image_uint8)
    image_pil = Image.fromarray(image_uint8)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    return buffer.getvalue()

# Affichage du contenu selon la page sélectionnée
if page == "Accueil":
    st.title("Bienvenue sur le projet Vision !")
    st.markdown("""
    Cette application vous permet de :
    - Visualiser et traiter des images
    - Tester différentes méthodes de binarisation
    - Rechercher des images dans la base
    """)
    st.image("https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=800&q=80", use_column_width=True)
    if image_path:
        st.subheader("Image sélectionnée :")
        st.code(image_path)
        st.image(Image.open(image_path), use_column_width=True)

elif page == "Traitement d'image":
    st.title("Traitement d'image")

    source_image = st.radio(
        "Source de l'image",
        ["Image de la base", "Importer une image"],
        horizontal=True,
    )

    fichier_uploade = None
    if source_image == "Importer une image":
        fichier_uploade = st.file_uploader(
            "Choisir une image",
            type=["jpg", "jpeg", "png", "bmp"],
        )

    image_traitement_pil = charger_image_traitement(
        image_path if source_image == "Image de la base" else None,
        fichier_uploade,
    )

    if image_traitement_pil is None:
        st.warning("Veuillez sélectionner ou importer une image pour commencer.")
    else:
        image_np_rgb = np.array(image_traitement_pil, dtype=np.uint8)
        image_np_gris = rgb_vers_gris_bt709(image_np_rgb).astype(np.float32)

        #st.subheader("Image source")
        #st.image(image_traitement_pil, use_column_width=True)

        st.markdown("---")
        operation = st.selectbox(
            "Opération",
            [
                "Conversion d'espace de couleur",
                "Filtre",
                "Ajout de bruit",
                "Augmentation pixel",
                "Padding",
            ],
        )

        resultat = None
        nom_resultat = "resultat_traitement.png"

        if operation == "Conversion d'espace de couleur":
            espace_cible = st.selectbox(
                "Espace de couleur cible",
                [
                    "Gris (moyenne)",
                    "Gris (BT.601)",
                    "Gris (BT.709)",
                    "HSV",
                    "HLS",
                    "YCrCb",
                    "Lab",
                    "Luv",
                    "XYZ",
                    "BGR",
                    "Image indexée",
                ],
            )

            nb_couleurs_indexees = 256
            if espace_cible == "Image indexée":
                nb_couleurs_indexees = st.slider(
                    "Nombre de couleurs indexées",
                    min_value=2,
                    max_value=256,
                    value=64,
                    step=1,
                )

            if st.button("Convertir l'image"):
                if espace_cible == "Gris (moyenne)":
                    resultat = rgb_vers_gris_moyenne(image_np_rgb)
                elif espace_cible == "Gris (BT.601)":
                    resultat = rgb_vers_gris_bt601(image_np_rgb)
                elif espace_cible == "Gris (BT.709)":
                    resultat = rgb_vers_gris_bt709(image_np_rgb)
                elif espace_cible == "HSV":
                    resultat = rgb_vers_hsv(image_np_rgb)
                elif espace_cible == "HLS":
                    resultat = rgb_vers_hls(image_np_rgb)
                elif espace_cible == "YCrCb":
                    resultat = rgb_vers_ycrcb(image_np_rgb)
                elif espace_cible == "Lab":
                    resultat = rgb_vers_lab(image_np_rgb)
                elif espace_cible == "Luv":
                    resultat = rgb_vers_luv(image_np_rgb)
                elif espace_cible == "XYZ":
                    resultat = rgb_vers_xyz(image_np_rgb)
                elif espace_cible == "BGR":
                    resultat = rgb_vers_bgr(image_np_rgb)
                else:
                    indices, palette = rgb_vers_image_indexee(image_np_rgb, nb_couleurs=nb_couleurs_indexees)
                    denom = float(max(nb_couleurs_indexees - 1, 1))
                    resultat = np.clip(indices.astype(np.float32) / denom * 255.0, 0, 255).astype(np.uint8)

                    palette_preview = np.tile(palette[np.newaxis, :, :], (24, 1, 1))
                    st.markdown("**Palette de l'image indexée**")
                    st.image(palette_preview, use_column_width=True)

                if espace_cible not in ["Gris (moyenne)", "Gris (BT.601)", "Gris (BT.709)", "Image indexée"]:
                    st.info("Affichage de l'image convertie avec ses canaux bruts (pas reconvertie en RGB).")

                nom_resultat = f"conversion_{espace_cible.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}.png"

        elif operation == "Filtre":
            type_filtre = st.selectbox(
                "Type de filtre",
                ["Prewitt", "Sobel", "Roberts", "Laplacien", "Scharr", "Aléatoire"],
            )
            methode = st.selectbox("Méthode", ["Convolution", "Cross-corrélation"])
            stride = st.slider("Stride", min_value=1, max_value=5, value=1, step=1)

            taille_filtre = None
            if type_filtre == "Aléatoire":
                taille_filtre = st.slider("Taille du filtre", min_value=3, max_value=11, value=3, step=2)

            if st.button("Appliquer le filtre"):
                operation_filtrage = convolution if methode == "Convolution" else cross_correlation

                if type_filtre == "Prewitt":
                    fx, fy = filtre_prewitt()
                    gx = operation_filtrage(image_np_gris, fx, stride=stride)
                    gy = operation_filtrage(image_np_gris, fy, stride=stride)
                    resultat = np.sqrt(gx**2 + gy**2)
                elif type_filtre == "Sobel":
                    fx, fy = filtre_sobel()
                    gx = operation_filtrage(image_np_gris, fx, stride=stride)
                    gy = operation_filtrage(image_np_gris, fy, stride=stride)
                    resultat = np.sqrt(gx**2 + gy**2)
                elif type_filtre == "Roberts":
                    fx, fy = filtre_roberts()
                    gx = operation_filtrage(image_np_gris, fx, stride=stride)
                    gy = operation_filtrage(image_np_gris, fy, stride=stride)
                    resultat = np.sqrt(gx**2 + gy**2)
                elif type_filtre == "Scharr":
                    fx, fy = filtre_scharr()
                    gx = operation_filtrage(image_np_gris, fx, stride=stride)
                    gy = operation_filtrage(image_np_gris, fy, stride=stride)
                    resultat = np.sqrt(gx**2 + gy**2)
                elif type_filtre == "Laplacien":
                    f = filtre_laplacien()
                    resultat = operation_filtrage(image_np_gris, f, stride=stride)
                else:
                    f = creation_filtre_aleatoire(taille_filtre)
                    resultat = operation_filtrage(image_np_gris, f, stride=stride)

                resultat = normaliser_pour_affichage(resultat)
                nom_resultat = f"filtre_{type_filtre.lower()}.png"

        elif operation == "Ajout de bruit":
            type_bruit = st.selectbox("Type de bruit", ["Gaussien", "Poivre et sel", "Poisson"])

            mean = 0
            sigma = 20
            proportion = 0.05

            if type_bruit == "Gaussien":
                mean = st.slider("Moyenne", min_value=-50, max_value=50, value=0, step=1)
                sigma = st.slider("Sigma", min_value=1, max_value=100, value=20, step=1)
            elif type_bruit == "Poivre et sel":
                proportion = st.slider("Proportion", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

            if st.button("Appliquer le bruit"):
                if type_bruit == "Gaussien":
                    resultat = ajouter_bruit_gaussien(image_np_rgb, mean=mean, sigma=sigma)
                elif type_bruit == "Poivre et sel":
                    resultat = ajouter_bruit_poivre_sel(image_np_rgb, proportion=proportion)
                else:
                    resultat = ajouter_bruit_poisson(image_np_rgb)

                nom_resultat = f"bruit_{type_bruit.lower().replace(' ', '_')}.png"

        elif operation == "Augmentation pixel":
            augmentation = st.selectbox(
                "Transformation pixel",
                [
                    "Transformation photométrique",
                    "Changement de la luminosité",
                    "Changement de contraste",
                    "Changement de gamma",
                    "Image renversée",
                ],
            )

            if augmentation == "Transformation photométrique":
                alpha = st.slider("Alpha (gain)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
                beta = st.slider("Beta (biais)", min_value=-100, max_value=100, value=0, step=1)

                if st.button("Appliquer la transformation"):
                    resultat = np.clip(alpha * image_np_rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)
                    nom_resultat = "augmentation_photometrique.png"

            elif augmentation == "Changement de la luminosité":
                delta = st.slider("Variation de luminosité", min_value=-100, max_value=100, value=20, step=1)

                if st.button("Appliquer la transformation"):
                    resultat = np.clip(image_np_rgb.astype(np.float32) + delta, 0, 255).astype(np.uint8)
                    nom_resultat = "augmentation_luminosite.png"

            elif augmentation == "Changement de contraste":
                facteur = st.slider("Facteur de contraste", min_value=0.1, max_value=3.0, value=1.2, step=0.1)

                if st.button("Appliquer la transformation"):
                    centre = 127.5
                    resultat = np.clip((image_np_rgb.astype(np.float32) - centre) * facteur + centre, 0, 255).astype(np.uint8)
                    nom_resultat = "augmentation_contraste.png"

            elif augmentation == "Changement de gamma":
                gamma = st.slider("Gamma", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

                if st.button("Appliquer la transformation"):
                    image_norm = image_np_rgb.astype(np.float32) / 255.0
                    resultat = np.clip((image_norm ** gamma) * 255.0, 0, 255).astype(np.uint8)
                    nom_resultat = "augmentation_gamma.png"

            else:
                type_renversement = st.selectbox(
                    "Type de renversement",
                    ["Horizontal", "Vertical", "Horizontal + Vertical"],
                )

                if st.button("Appliquer la transformation"):
                    if type_renversement == "Horizontal":
                        resultat = np.fliplr(image_np_rgb)
                    elif type_renversement == "Vertical":
                        resultat = np.flipud(image_np_rgb)
                    else:
                        resultat = np.flipud(np.fliplr(image_np_rgb))
                    nom_resultat = "augmentation_renversement.png"

        else:
            mode_padding = st.selectbox("Mode de padding", ["constant", "edge", "reflect"])
            pad_y = st.slider("Padding vertical", min_value=0, max_value=100, value=20, step=1)
            pad_x = st.slider("Padding horizontal", min_value=0, max_value=100, value=20, step=1)
            valeur_constante = 0
            if mode_padding == "constant":
                valeur_constante = st.slider("Valeur constante", min_value=0, max_value=255, value=0, step=1)

            if st.button("Appliquer le padding"):
                resultat = ajouter_padding(
                    image_np_rgb,
                    pad_y=pad_y,
                    pad_x=pad_x,
                    mode=mode_padding,
                    valeur=valeur_constante,
                )
                nom_resultat = f"padding_{mode_padding}.png"

        if resultat is not None:
            st.subheader("Comparaison")
            col_avant, col_apres = st.columns(2)
            with col_avant:
                st.markdown("**Image de départ**")
                st.image(image_np_rgb, use_column_width=True)
            with col_apres:
                st.markdown("**Image modifiée**")
                st.image(resultat, use_column_width=True)
            st.download_button(
                label="Télécharger le résultat",
                data=image_array_vers_bytes_png(resultat),
                file_name=nom_resultat,
                mime="image/png",
            )

elif page == "Moteur de recherche":
    st.title("Moteur de recherche d'images")

    if image_path:
        st.subheader("Image sélectionnée :")
        st.code(image_path)
        st.image(Image.open(image_path), width=300)

    st.markdown("---")
    st.subheader("Paramètres de recherche")

    col1, col2, col3, col4 = st.columns(4)

    DESCRIPTEURS_COULEUR_QUANTIFICATION = [
        "Histogramme image indexée",
        "Entropie",
        "Image indexée",
        "Histogramme bloc",
        "CSV",
        "DCD",
        "CCD",
    ]
    DESCRIPTEURS_TEXTURE = [
        "Matrice de concurrence",
        "Histogramme bloc LBP",
        "LBP",
    ]
    DESCRIPTEURS_FORME = [
        "Histogramme de direction de gradient",
        "Histogramme pondéré par la norme",
        "Histogramme de bloc (direction de gradient)",
    ]

    DESCRIPTEURS_TEXTURE_FORME = DESCRIPTEURS_TEXTURE + DESCRIPTEURS_FORME

    DESCRIPTEURS_GRIS = [
        "Histogramme niveaux de gris",
        "Entropie",
        "Image indexée",
        "Histogramme bloc",
    ] + DESCRIPTEURS_TEXTURE_FORME

    DESCRIPTEURS_PAR_ESPACE = {
        "RGB":            ["Histogramme couleur RGB"] + DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "HSV":            ["Histogramme HSV"] + DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "HLS":            DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "YCrCb":          DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "Lab":            DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "Luv":            DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "XYZ":            DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
        "Gris (moyenne)": DESCRIPTEURS_GRIS,
        "Gris (BT.601)":  DESCRIPTEURS_GRIS,
        "Gris (BT.709)":  DESCRIPTEURS_GRIS,
        "BGR":            ["Histogramme couleur RGB"] + DESCRIPTEURS_COULEUR_QUANTIFICATION + DESCRIPTEURS_TEXTURE_FORME,
    }

    with col1:
        espace_couleur = st.selectbox(
            "Espace de couleur",
            [
                "RGB",
                "HSV",
                "HLS",
                "YCrCb",
                "Lab",
                "Luv",
                "XYZ",
                "Gris (moyenne)",
                "Gris (BT.601)",
                "Gris (BT.709)",
                "BGR",
            ]
        )

    with col2:
        descripteurs_disponibles = DESCRIPTEURS_PAR_ESPACE.get(espace_couleur, ["Histogramme couleur RGB"])
        descripteur = st.selectbox("Descripteur", descripteurs_disponibles)

    with col3:
        distance = st.selectbox(
            "Distance",
            [
                "L1 (Manhattan)",
                "L2 (Euclidienne)",
                "Cosinus",
                "Chi²",
                "Intersection",
            ]
        )

    with col4:
        n_resultats = st.number_input("Nombre de résultats (n)", min_value=1, max_value=100, value=10, step=1)
        
        
    ESPACE_COULEUR_MAP = {
        "RGB":            "rgb",
        "HSV":            "hsv",
        "HLS":            "hls",
        "YCrCb":          "ycrcb",
        "Lab":            "lab",
        "Luv":            "luv",
        "XYZ":            "xyz",
        "Gris (moyenne)": "gris_moyenne",
        "Gris (BT.601)":  "gris_bt601",
        "Gris (BT.709)":  "gris_bt709",
        "BGR":            "bgr",
    }
    DESCRIPTEUR_MAP = {
        "Histogramme niveaux de gris": "gris",
        "Histogramme couleur RGB":     "couleur",
        "Histogramme HSV":             "hsv",
        "Histogramme image indexée":   "indexe",
        "Entropie":                    "entropie",
        "Image indexée":               "image_indexee",
        "Histogramme bloc":            "hist_bloc",
        "CSV":                         "csv",
        "DCD":                         "dcd",
        "CCD":                         "ccd",
        "Matrice de concurrence":      "glcm",
        "Histogramme bloc LBP":        "lbp_bloc",
        "LBP":                         "lbp",
        "Histogramme de direction de gradient":        "grad_dir",
        "Histogramme pondéré par la norme":            "grad_mag",
        "Histogramme de bloc (direction de gradient)": "grad_bloc",
    }
    DISTANCE_MAP = {
        "L1 (Manhattan)":  "l1",
        "L2 (Euclidienne)": "l2",
        "Cosinus":          "cosinus",
        "Chi²":             "chi2",
        "Intersection":     "intersection",
    }

    DESCRIPTEUR_FN_MAP = {
        "gris":     lambda img: calculer_histogramme_gris(img, bins=256, normaliser=True),
        "couleur":  lambda img: calculer_histogramme_couleur(img, bins=256, normaliser=True),
        "hsv":      lambda img: calculer_histogramme_hsv(img, bins_h=180, bins_s=256, bins_v=256, normaliser=True),
        "indexe":   lambda img: calculer_histogramme_indexe(img, nb_couleurs=256, normaliser=True),
        "entropie": lambda img: calculer_entropie(img, bins=256),
        "image_indexee": lambda img: calculer_image_indexee_descripteur(img, nb_couleurs=64),
        "hist_bloc":     lambda img: calculer_histogramme_bloc(img, bins=32, taille_bloc=(32, 32), normaliser=True),
        "csv":           lambda img: calculer_csv(img, nb_couleurs=64),
        "dcd":           lambda img: calculer_dcd(img, nb_couleurs=64, top_k=8),
        "ccd":           lambda img: calculer_ccd(img, nb_couleurs=32, seuil_coherence=30),
        "glcm":          lambda img: calculer_matrice_concurrence(img, niveaux=16, distances=(1,), angles=(0,), normaliser=True),
        "lbp_bloc":      lambda img: calculer_histogramme_bloc_lbp(img, P=8, R=1, method='uniform', taille_bloc=(32, 32)),
        "lbp":           lambda img: calculer_lbp(img, P=8, R=1, method='uniform'),
        "grad_dir":      lambda img: calculer_histogramme_direction_gradient(img, bins=9, normaliser=True),
        "grad_mag":      lambda img: calculer_histogramme_pondere_par_norme(img, bins=9, normaliser=True),
        "grad_bloc":     lambda img: calculer_histogramme_bloc_direction_gradient(img, bins=9, taille_bloc=(16, 16), normaliser=True),
    }
    DISTANCE_FN_MAP = {
        "l1":           distance_l1,
        "l2":           distance_l2,
        "cosinus":      distance_cosinus,
        "chi2":         distance_chi2,
        "intersection": distance_intersection,
    }

    st.markdown("---")
    if not image_path:
        st.warning("Veuillez sélectionner une image dans la barre latérale.")
    else:
        espace_key = ESPACE_COULEUR_MAP[espace_couleur]
        cache_dir = os.path.join("cache_map")
        os.makedirs(cache_dir, exist_ok=True)
        map_table_path = os.path.join(cache_dir, f"map_table_{espace_key}.json")

        descripteurs_keys_espace = [DESCRIPTEUR_MAP[d] for d in descripteurs_disponibles]
        descripteur_fn_subset = {
            k: DESCRIPTEUR_FN_MAP[k]
            for k in descripteurs_keys_espace
            if k in DESCRIPTEUR_FN_MAP
        }

        distance_label_rev = {
            "l1": "L1 (Manhattan)",
            "l2": "L2 (Euclidienne)",
            "cosinus": "Cosinus",
            "chi2": "Chi²",
            "intersection": "Intersection",
        }
        descripteur_label_rev = {
            v: k for k, v in DESCRIPTEUR_MAP.items()
        }

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            lancer_recherche = st.button("Lancer la recherche")
        with col_btn2:
            calculer_toutes_map = st.button("Calculer toutes les MAP")
        with col_btn3:
            afficher_tableau = st.button("Afficher tableau")

        if lancer_recherche:
            with st.spinner("Recherche en cours..."):
                resultats, ap, fig = afficher_images_proches(
                    image_path=image_path,
                    dossier_base=DOSSIER_BASE,
                    n=int(n_resultats),
                    descripteur=DESCRIPTEUR_MAP[descripteur],
                    distance=DISTANCE_MAP[distance],
                    espace_couleur=espace_key,
                )
            st.session_state["recherche_ap"]  = ap
            st.session_state["recherche_fig"] = fig

            desc_key = DESCRIPTEUR_MAP[descripteur]
            dist_key = DISTANCE_MAP[distance]
            with st.spinner("Calcul du MAP sur toute la base (peut prendre plusieurs minutes)..."):
                map_score = calculer_map(
                    dossier_base=DOSSIER_BASE,
                    espace_couleur=espace_key,
                    descripteur_fn=DESCRIPTEUR_FN_MAP[desc_key],
                    distance_fn=DISTANCE_FN_MAP[dist_key],
                )
            st.session_state["map_score"]  = map_score
            st.session_state["map_label"]  = f"Espace: {espace_couleur}  |  Descripteur: {descripteur}  |  Distance: {distance}"

        if calculer_toutes_map:
            if os.path.exists(map_table_path):
                with open(map_table_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                st.session_state["map_table_payload"] = payload
                st.info(f"Résultats déjà calculés pour {espace_couleur}. Fichier rechargé: {map_table_path}")
            else:
                with st.spinner("Calcul des MAP pour toutes les combinaisons (première fois seulement)..."):
                    lignes_map = calculer_map_toutes_combinaisons(
                        dossier_base=DOSSIER_BASE,
                        espace_couleur=espace_key,
                        descripteur_fn_map=descripteur_fn_subset,
                        distance_fn_map=DISTANCE_FN_MAP,
                    )

                payload = {
                    "espace_label": espace_couleur,
                    "espace_key": espace_key,
                    "nb_descripteurs": len(descripteur_fn_subset),
                    "nb_distances": len(DISTANCE_FN_MAP),
                    "rows": lignes_map,
                }

                with open(map_table_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                st.session_state["map_table_payload"] = payload
                st.success(f"Calcul terminé et sauvegardé: {map_table_path}")

        if afficher_tableau:
            payload = st.session_state.get("map_table_payload")

            if payload is None:
                if os.path.exists(map_table_path):
                    with open(map_table_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    st.session_state["map_table_payload"] = payload
                else:
                    st.warning("Aucun résultat sauvegardé pour cet espace. Lance d'abord 'Calculer toutes les MAP'.")

            if payload is not None:
                rows = payload.get("rows", [])
                table_rows = [
                    {
                        "Descripteur": descripteur_label_rev.get(r["descripteur"], r["descripteur"]),
                        "Distance": distance_label_rev.get(r["distance"], r["distance"]),
                        "MAP": round(float(r["map"]), 4),
                    }
                    for r in rows
                ]

                st.subheader(f"Tableau MAP - {payload.get('espace_label', espace_couleur)}")
                st.dataframe(table_rows, use_container_width=True)

                if table_rows:
                    st.markdown("**Top 5 combinaisons**")
                    st.dataframe(table_rows[:5], use_container_width=True)

        # Affichage persistant des résultats
        if "recherche_ap" in st.session_state:
            #st.success(f"AP (Average Precision) pour cette image : **{st.session_state['recherche_ap']:.4f}**")
            st.pyplot(st.session_state["recherche_fig"])

        if "map_score" in st.session_state:
            st.success(f"MAP global : **{st.session_state['map_score']:.4f}**  |  {st.session_state['map_label']}")

elif page == "Binarisation":
    st.title("Binarisation et Segmentation d'images")

    source_binarisation = st.radio(
        "Source de l'image",
        ["Image de la base", "Importer une image"],
        horizontal=True,
    )

    fichier_binarisation = None
    if source_binarisation == "Importer une image":
        fichier_binarisation = st.file_uploader(
            "Choisir une image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="binarisation_uploader",
        )

    image_binarisation_pil = charger_image_traitement(
        image_path if source_binarisation == "Image de la base" else None,
        fichier_binarisation,
    )

    if image_binarisation_pil is None:
        st.warning("Veuillez sélectionner ou importer une image pour commencer.")
    else:
        image_bin_rgb = np.array(image_binarisation_pil, dtype=np.uint8)

        st.markdown("---")
        st.subheader("Sélectionner une technique")

        # Organiser les techniques par catégories
        techniques = {
            "🔲 Niveaux de Gris (Globales)": [
                "Otsu (automatique)",
                "Moyenne",
                "Médiane",
                "Min-Max (moyenne)",
                "Écart-type",
                "P-tile (50%)",
                "Moyenne tronquée",
            ],
            "🔍 Teinte/Hue (Globales)": [
                "Teinte (Otsu)",
                "Hue - Moyenne",
                "Hue - Médiane",
                "Hue - MinMax",
                "Hue - Écart-type",
                "Hue - P-tile",
                "Hue - Moyenne tronquée",
            ],
            "♻️ ISODATA": [
                "ISODATA (init. Moyenne)",
                "ISODATA (init. 4 Coins)",
            ],
            "🔸 Locales (Adaptatives fenêtre)": [
                "Locale - Moyenne",
                "Locale - Médiane",
                "Locale - Min-Max",
            ],
            "⚙️ Adaptative (OpenCV optimisée)": [
                "Adaptative (Gaussian)",
            ],
            "🎨 Segmentation": [
                "K-Means",
                "Up Sampling",
                "FCN",
                "U-Net",
                "SegNet",
                "PSPNet",
            ],
        }

        # Créer un selectbox avec les catégories
        categorie_selectionnee = st.selectbox("Catégorie", list(techniques.keys()))
        technique = st.selectbox("Technique", techniques[categorie_selectionnee])

        col_tech, col_params = st.columns([1, 1])

        # NIVEAUX DE GRIS - GLOBALES
        if technique == "Otsu (automatique)":
            if st.button("Appliquer Otsu"):
                gray, binary, hist, seuil = binarisation_niveaux_gris(image_bin_rgb, methode='otsu')
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Otsu (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}% | **Noirs** : {stats['pixels_noirs_pct']:.1f}%")

        elif technique == "Moyenne":
            if st.button("Appliquer"):
                gray, binary, seuil = binarisation_gris_moyenne(image_bin_rgb)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Moyenne (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}%")

        elif technique == "Médiane":
            if st.button("Appliquer"):
                gray, binary, seuil = binarisation_gris_mediane(image_bin_rgb)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Médiane (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}%")

        elif technique == "Min-Max (moyenne)":
            if st.button("Appliquer"):
                gray, binary, seuil = binarisation_gris_minmax(image_bin_rgb)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**(Min+Max)/2 (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}%")

        elif technique == "Écart-type":
            if st.button("Appliquer"):
                gray, binary, seuil = binarisation_gris_ecart_type(image_bin_rgb)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Écart-type (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}%")

        elif technique == "P-tile (50%)":
            if st.button("Appliquer"):
                gray, binary, seuil = binarisation_gris_ptile(image_bin_rgb, percentile=50)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**P-tile 50% (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}%")

        elif technique == "Moyenne tronquée":
            if st.button("Appliquer"):
                gray, binary, seuil = binarisation_gris_moyenne_tronquee(image_bin_rgb, pourcentage=10)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Moyenne tronquée (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Statistiques"):
                    st.write(f"**Seuil** : {seuil} | **Blancs** : {stats['pixels_blancs_pct']:.1f}%")

        # TEINTE/HUE - GLOBALES
        elif technique == "Teinte (Otsu)":
            if st.button("Appliquer"):
                hsv, h_channel, binary, hist = binarisation_teinte(image_bin_rgb)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown("**Binarisation Teinte (Otsu)**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Hue - Moyenne":
            if st.button("Appliquer"):
                h_channel, binary, seuil = binarisation_hue_moyenne(image_bin_rgb)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Hue - Moyenne (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Hue - Médiane":
            if st.button("Appliquer"):
                h_channel, binary, seuil = binarisation_hue_mediane(image_bin_rgb)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Hue - Médiane (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Hue - MinMax":
            if st.button("Appliquer"):
                h_channel, binary, seuil = binarisation_hue_minmax(image_bin_rgb)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Hue - MinMax (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Hue - Écart-type":
            if st.button("Appliquer"):
                h_channel, binary, seuil = binarisation_hue_ecart_type(image_bin_rgb)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Hue - Écart-type (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Hue - P-tile":
            if st.button("Appliquer"):
                h_channel, binary, seuil = binarisation_hue_ptile(image_bin_rgb, percentile=50)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Hue - P-tile (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Hue - Moyenne tronquée":
            if st.button("Appliquer"):
                h_channel, binary, seuil = binarisation_hue_moyenne_tronquee(image_bin_rgb, pourcentage=10)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Hue - Moyenne tronquée (seuil={seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        # ISODATA
        elif technique == "ISODATA (init. Moyenne)":
            if st.button("Appliquer"):
                gray, binary, seuil, iterations, init = binarisation_isodata_moyenne(image_bin_rgb)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**ISODATA (init={init}→{seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Détails"):
                    st.write(f"**Seuil initial** : {init} | **Seuil final** : {seuil} | **Itérations** : {iterations}")

        elif technique == "ISODATA (init. 4 Coins)":
            if st.button("Appliquer"):
                gray, binary, seuil, iterations, init = binarisation_isodata_4coins(image_bin_rgb)
                stats = calculer_statistiques_binarisation(gray, binary, seuil)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**ISODATA 4 Coins (init={init:.0f}→{seuil})**")
                    st.image(binary, use_column_width=True, channels="GRAY")
                with st.expander("Détails"):
                    st.write(f"**Seuil initial (4 coins)** : {init:.0f} | **Seuil final** : {seuil} | **Itérations** : {iterations}")

        # LOCALES
        elif technique == "Locale - Moyenne":
            with col_params:
                taille_fenetre = st.slider("Taille fenêtre", 3, 31, 7, step=2)
            if st.button("Appliquer"):
                gray, binary = binarisation_locale_moyenne(image_bin_rgb, taille_fenetre=taille_fenetre)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Locale - Moyenne (fenêtre={taille_fenetre})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Locale - Médiane":
            with col_params:
                taille_fenetre = st.slider("Taille fenêtre", 3, 31, 7, step=2)
            if st.button("Appliquer"):
                gray, binary = binarisation_locale_mediane(image_bin_rgb, taille_fenetre=taille_fenetre)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Locale - Médiane (fenêtre={taille_fenetre})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        elif technique == "Locale - Min-Max":
            with col_params:
                taille_fenetre = st.slider("Taille fenêtre", 3, 31, 7, step=2)
            if st.button("Appliquer"):
                with st.spinner("Calcul en cours (peut prendre du temps)..."):
                    gray, binary = binarisation_locale_minmax(image_bin_rgb, taille_fenetre=taille_fenetre)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Locale - Min-Max (fenêtre={taille_fenetre})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        # ADAPTATIVE
        elif technique == "Adaptative (Gaussian)":
            with col_params:
                block_size = st.slider("Taille bloc", 3, 31, 11, step=2)
            if st.button("Appliquer"):
                gray, binary = binarisation_adaptatif(image_bin_rgb, block_size=block_size)
                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Adaptative Gaussian (bloc={block_size})**")
                    st.image(binary, use_column_width=True, channels="GRAY")

        # K-MEANS
        elif technique == "K-Means":
            col_k, col_space = st.columns(2)
            with col_k:
                k_clusters = st.slider("K", 2, 8, 3)
            with col_space:
                espace_kmeans = st.selectbox("Espace", ["RGB", "HSV", "Gris"])
            
            if st.button("Appliquer K-Means"):
                espace_map = {"RGB": "rgb", "HSV": "hsv", "Gris": "gris"}
                seg_img, labels, centres, inertia = segmentation_kmeans(
                    image_bin_rgb, k=k_clusters, espace_couleur=espace_map[espace_kmeans], random_state=42
                )
                if len(seg_img.shape) == 2:
                    seg_display = seg_img
                else:
                    seg_display = seg_img.astype(np.uint8)

                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**K-Means (K={k_clusters}, {espace_kmeans})**")
                    if len(seg_display.shape) == 2:
                        st.image(seg_display, use_column_width=True, channels="GRAY")
                    else:
                        st.image(seg_display, use_column_width=True)
                with st.expander("Informations"):
                    st.write(f"**Clusters** : {k_clusters} | **Espace** : {espace_kmeans} | **Inertie** : {inertia:.2f}")

        elif technique == "Up Sampling":
            col_facteur, col_methode = st.columns(2)
            with col_facteur:
                facteur = st.slider("Facteur d'agrandissement", 2, 6, 2)
            with col_methode:
                methode_up = st.selectbox("Interpolation", ["bilinear", "nearest", "bicubic"])

            if st.button("Appliquer Up Sampling"):
                image_up = up_sampling(image_bin_rgb, facteur=facteur, methode=methode_up)

                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown(f"**Up Sampling x{facteur} ({methode_up})**")
                    st.image(image_up, use_column_width=True)

        elif technique == "FCN":
            if st.button("Appliquer FCN"):
                with st.spinner("Segmentation FCN en cours..."):
                    mask, image_colorisee, info = segmentation_fcn(image_bin_rgb)

                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown("**FCN - Masque colorisé**")
                    st.image(image_colorisee, use_column_width=True)
                with st.expander("Informations"):
                    st.write(info)

        elif technique == "U-Net":
            if st.button("Appliquer U-Net"):
                with st.spinner("Segmentation U-Net en cours..."):
                    mask, image_colorisee, info = segmentation_unet(image_bin_rgb)

                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown("**U-Net - Masque colorisé**")
                    st.image(image_colorisee, use_column_width=True)
                with st.expander("Informations"):
                    st.write(info)

        elif technique == "SegNet":
            if st.button("Appliquer SegNet"):
                with st.spinner("Segmentation SegNet en cours..."):
                    mask, image_colorisee, info = segmentation_segnet(image_bin_rgb)

                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown("**SegNet - Masque colorisé**")
                    st.image(image_colorisee, use_column_width=True)
                with st.expander("Informations"):
                    st.write(info)

        elif technique == "PSPNet":
            if st.button("Appliquer PSPNet"):
                with st.spinner("Segmentation PSPNet en cours..."):
                    mask, image_colorisee, info = segmentation_pspnet(image_bin_rgb)

                st.subheader("Résultats")
                col_avant, col_apres = st.columns(2)
                with col_avant:
                    st.markdown("**Image de départ**")
                    st.image(image_bin_rgb, use_column_width=True)
                with col_apres:
                    st.markdown("**PSPNet - Masque colorisé**")
                    st.image(image_colorisee, use_column_width=True)
                with st.expander("Informations"):
                    st.write(info)