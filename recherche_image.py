import os
import re
import numpy as np
from tqdm import tqdm
import cv2

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


def charger_base_images(dossier_base, espace_couleur='rgb'):
    """
    Charge toutes les images d'un dossier et les convertit dans l'espace de couleur choisi.
    
    Paramètres:
    -----------
    dossier_base : str
        Chemin vers le dossier contenant les images
    espace_couleur : str
        Espace de couleur cible : 'rgb', 'hsv', 'hls', 'ycrcb', 'lab', 'luv', 'xyz',
        'gris_moyenne', 'gris_bt601', 'gris_bt709', 'bgr'
    
    Retourne:
    ---------
    dict : Dictionnaire {chemin_image: image_convertie}
    """
    import os
    import glob
    from tqdm import tqdm
    
    if not os.path.exists(dossier_base):
        raise FileNotFoundError(f"Dossier introuvable: {dossier_base}")
    
    # Récupérer tous les fichiers images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    fichiers_images = []
    for ext in extensions:
        fichiers_images.extend(glob.glob(os.path.join(dossier_base, '**', ext), recursive=True))
    
    print(f"\n{'='*80}")
    print(f"📁 CHARGEMENT DE LA BASE D'IMAGES")
    print(f"{'='*80}")
    print(f"📂 Dossier : {dossier_base}")
    print(f"🎨 Espace de couleur : {espace_couleur.upper()}")
    print(f"📷 Nombre d'images trouvées : {len(fichiers_images)}")
    print(f"{'─'*80}\n")
    
    if len(fichiers_images) == 0:
        print("⚠️ Aucune image trouvée!")
        return {}
    
    # Dictionnaire de conversion
    conversions = {
        'rgb': lambda img: img,
        'hsv': rgb_vers_hsv,
        'hls': rgb_vers_hls,
        'ycrcb': rgb_vers_ycrcb,
        'lab': rgb_vers_lab,
        'luv': rgb_vers_luv,
        'xyz': rgb_vers_xyz,
        'gris_moyenne': rgb_vers_gris_moyenne,
        'gris_bt601': rgb_vers_gris_bt601,
        'gris_bt709': rgb_vers_gris_bt709,
        'bgr': rgb_vers_bgr,
    }
    
    if espace_couleur.lower() not in conversions:
        raise ValueError(f"Espace de couleur inconnu: {espace_couleur}. "
                        f"Disponibles: {list(conversions.keys())}")
    
    fonction_conversion = conversions[espace_couleur.lower()]
    
    # Charger et convertir les images
    base_images = {}
    erreurs = 0
    
    for img_path in tqdm(fichiers_images, desc="Chargement et conversion"):
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                erreurs += 1
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_convertie = fonction_conversion(img_rgb)
            base_images[img_path] = img_convertie
            
        except Exception as e:
            erreurs += 1
            print(f"⚠️ Erreur avec {img_path}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Chargement terminé!")
    print(f"   Images chargées avec succès : {len(base_images)}")
    print(f"   Erreurs : {erreurs}")
    print(f"{'='*80}\n")
    
    return base_images

def calculer_map(dossier_base, espace_couleur, descripteur_fn, distance_fn):
    """
    Calcule le MAP (Mean Average Precision) sur toute la base d'images.

    Paramètres:
    -----------
    dossier_base   : str      - Chemin vers le dossier de la base d'images
    espace_couleur : str      - Espace de couleur ('rgb', 'gris_bt601', 'hsv', ...)
    descripteur_fn : callable - Fonction de calcul du descripteur (ex: calculer_histogramme_gris)
    distance_fn    : callable - Fonction de distance (ex: distance_l2)

    Retourne:
    ---------
    float : Score MAP global
    """
    def extract_image_number(path):
        match = re.search(r'image(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else float('inf')

    base = charger_base_images(dossier_base, espace_couleur=espace_couleur)
    liste_images = sorted(base.keys(), key=extract_image_number)

    # Pré-calcul des descripteurs
    descripteurs = {}
    for path in tqdm(liste_images, desc="Pré-calcul des descripteurs"):
        descripteurs[path] = descripteur_fn(base[path])

    scores_moyens = []

    for image_path in tqdm(liste_images, desc="Calcul MAP"):
        classe_cible = os.path.basename(os.path.dirname(image_path))
        image_path_abs = os.path.abspath(image_path)
        desc_cible = descripteurs[image_path]

        resultats = [
            (k, distance_fn(desc_cible, v))
            for k, v in descripteurs.items()
            if os.path.abspath(k) != image_path_abs
        ]
        resultats.sort(key=lambda x: x[1])

        positions_meme_classe = [
            i for i, (chemin, _) in enumerate(resultats, 1)
            if os.path.basename(os.path.dirname(chemin)) == classe_cible
        ]

        precisions = [idx / pos for idx, pos in enumerate(positions_meme_classe, 1)]
        scores_moyens.append(np.mean(precisions) if precisions else 0.0)

    map_score = np.mean(scores_moyens) if scores_moyens else 0.0
    print(f"\n{'='*60}")
    print(f"MAP ({espace_couleur} | {descripteur_fn.__name__} | {distance_fn.__name__}) : {map_score:.4f}")
    print(f"{'='*60}")
    return map_score


def calculer_map_toutes_combinaisons(dossier_base, espace_couleur, descripteur_fn_map, distance_fn_map):
    """
    Calcule le MAP pour toutes les combinaisons descripteur x distance.
    Charge la base une seule fois et réutilise les descripteurs précalculés.

    Paramètres:
    -----------
    dossier_base : str
    espace_couleur : str
    descripteur_fn_map : dict[str, callable]
    distance_fn_map : dict[str, callable]

    Retourne:
    ---------
    list[dict] : [{"descripteur": ..., "distance": ..., "map": ...}, ...]
    """
    def extract_image_number(path):
        match = re.search(r'image(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else float('inf')

    base = charger_base_images(dossier_base, espace_couleur=espace_couleur)
    liste_images = sorted(base.keys(), key=extract_image_number)

    # 1) Pré-calculer une fois les descripteurs pour chaque méthode
    descripteurs_par_methode = {}
    for nom_desc, fn_desc in descripteur_fn_map.items():
        descripteurs = {}
        for path in tqdm(liste_images, desc=f"Descripteurs [{nom_desc}]"):
            descripteurs[path] = fn_desc(base[path])
        descripteurs_par_methode[nom_desc] = descripteurs

    resultats = []

    # 2) MAP pour chaque combinaison descripteur-distance
    for nom_desc, descripteurs in descripteurs_par_methode.items():
        for nom_dist, fn_dist in distance_fn_map.items():
            scores_moyens = []

            for image_path in tqdm(liste_images, desc=f"MAP [{nom_desc} x {nom_dist}]"):
                classe_cible = os.path.basename(os.path.dirname(image_path))
                image_path_abs = os.path.abspath(image_path)
                desc_cible = descripteurs[image_path]

                classements = [
                    (k, fn_dist(desc_cible, v))
                    for k, v in descripteurs.items()
                    if os.path.abspath(k) != image_path_abs
                ]
                classements.sort(key=lambda x: x[1])

                positions_meme_classe = [
                    i for i, (chemin, _) in enumerate(classements, 1)
                    if os.path.basename(os.path.dirname(chemin)) == classe_cible
                ]

                precisions = [idx / pos for idx, pos in enumerate(positions_meme_classe, 1)]
                scores_moyens.append(np.mean(precisions) if precisions else 0.0)

            map_score = float(np.mean(scores_moyens) if scores_moyens else 0.0)
            resultats.append({
                "descripteur": nom_desc,
                "distance": nom_dist,
                "map": map_score,
            })

            print(f"MAP ({espace_couleur} | {nom_desc} | {nom_dist}) : {map_score:.4f}")

    resultats.sort(key=lambda r: r["map"], reverse=True)
    return resultats

def afficher_images_proches(image_path, dossier_base, n=10, descripteur='couleur', distance='l2', espace_couleur='rgb'):
    """
    Affiche les n images les plus proches d'une image cible dans la base.

    Paramètres:
    -----------
    image_path : str
        Chemin vers l'image cible
    dossier_base : str
        Chemin vers le dossier racine de la base d'images
    n : int
        Nombre d'images proches à retourner et afficher (défaut: 10)
    descripteur : str
        'gris', 'couleur', 'hsv', 'indexe'
    distance : str
        'l1', 'l2', 'cosinus', 'chi2', 'intersection'
    espace_couleur : str
        'rgb', 'hsv', 'hls', 'ycrcb', 'lab', 'luv', 'xyz', 'gris_bt601', 'gris_bt709', 'gris_moyenne', 'bgr'

    Retourne:
    ---------
    tuple (resultats, precision_moyenne, fig)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    distance_map = {
        'l1':           distance_l1,
        'l2':           distance_l2,
        'cosinus':      distance_cosinus,
        'chi2':         distance_chi2,
        'intersection': distance_intersection,
    }

    descripteur_fn_map = {
        'gris':     lambda img: calculer_histogramme_gris(img, bins=256, normaliser=True),
        'couleur':  lambda img: calculer_histogramme_couleur(img, bins=256, normaliser=True),
        'hsv':      lambda img: calculer_histogramme_hsv(img, bins_h=180, bins_s=256, bins_v=256, normaliser=True),
        'indexe':   lambda img: calculer_histogramme_indexe(img, nb_couleurs=256, normaliser=True),
        'entropie': lambda img: calculer_entropie(img, bins=256),
        'image_indexee': lambda img: calculer_image_indexee_descripteur(img, nb_couleurs=64),
        'hist_bloc':     lambda img: calculer_histogramme_bloc(img, bins=32, taille_bloc=(32, 32), normaliser=True),
        'csv':           lambda img: calculer_csv(img, nb_couleurs=64),
        'dcd':           lambda img: calculer_dcd(img, nb_couleurs=64, top_k=8),
        'ccd':           lambda img: calculer_ccd(img, nb_couleurs=32, seuil_coherence=30),
        'glcm':          lambda img: calculer_matrice_concurrence(img, niveaux=16, distances=(1,), angles=(0,), normaliser=True),
        'lbp_bloc':      lambda img: calculer_histogramme_bloc_lbp(img, P=8, R=1, method='uniform', taille_bloc=(32, 32)),
        'lbp':           lambda img: calculer_lbp(img, P=8, R=1, method='uniform'),
        'grad_dir':      lambda img: calculer_histogramme_direction_gradient(img, bins=9, normaliser=True),
        'grad_mag':      lambda img: calculer_histogramme_pondere_par_norme(img, bins=9, normaliser=True),
        'grad_bloc':     lambda img: calculer_histogramme_bloc_direction_gradient(img, bins=9, taille_bloc=(16, 16), normaliser=True),
    }

    if descripteur not in descripteur_fn_map:
        raise ValueError(f"Descripteur inconnu: '{descripteur}'. Disponibles: {list(descripteur_fn_map.keys())}")
    if distance not in distance_map:
        raise ValueError(f"Distance inconnue: '{distance}'. Disponibles: {list(distance_map.keys())}")

    fn_distance    = distance_map[distance]
    fn_descripteur = descripteur_fn_map[descripteur]

    # Charger l'image cible
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    img_rgb_cible = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    conv_map = {
        'rgb':          lambda img: img,
        'hsv':          rgb_vers_hsv,
        'hls':          rgb_vers_hls,
        'ycrcb':        rgb_vers_ycrcb,
        'lab':          rgb_vers_lab,
        'luv':          rgb_vers_luv,
        'xyz':          rgb_vers_xyz,
        'bgr':          rgb_vers_bgr,
        'gris_bt601':   rgb_vers_gris_bt601,
        'gris_bt709':   rgb_vers_gris_bt709,
        'gris_moyenne': rgb_vers_gris_moyenne,
    }
    fn_display = conv_map.get(espace_couleur, lambda img: img)
    cmap_display = 'gray' if espace_couleur in ('gris_bt601', 'gris_bt709', 'gris_moyenne') else None
    img_cible = fn_display(img_rgb_cible)

    # Charger la base et exclure l'image cible
    base = charger_base_images(dossier_base, espace_couleur=espace_couleur)
    image_path_abs = os.path.abspath(image_path)
    base_sans_cible = {k: v for k, v in base.items() if os.path.abspath(k) != image_path_abs}

    # Calculer les distances
    desc_cible = fn_descripteur(img_cible)
    distances_list = []
    for chemin, img in base_sans_cible.items():
        dist = fn_distance(desc_cible, fn_descripteur(img))
        distances_list.append((chemin, dist))

    distances_list.sort(key=lambda x: x[1])
    resultats = distances_list[:n]

    # Affichage texte
    classe_cible = os.path.basename(os.path.dirname(image_path))

    print("\n" + "="*80)
    print("🔍 ANALYSE DES RÉSULTATS - POSITIONS DES IMAGES DE LA MÊME CLASSE")
    print("="*80)
    print(f"Image cible   : {os.path.basename(image_path)}")
    print(f"Classe cible  : {classe_cible}")
    print(f"Descripteur   : {descripteur.upper()}")
    print(f"Distance      : {distance.upper()}")
    print(f"Nb images (n) : {n}")
    print("─"*80)

    positions_meme_classe = []
    for i, (chemin, dist) in enumerate(resultats, 1):
        nom_fichier  = os.path.basename(chemin)
        classe_image = os.path.basename(os.path.dirname(chemin))
        est_meme_classe = classe_image == classe_cible
        marqueur = "✓" if est_meme_classe else "✗"
        if est_meme_classe:
            positions_meme_classe.append(i)
        print(f"{marqueur} Position {i:3d} | {nom_fichier:<40} | Classe: {classe_image:<40} | Distance: {dist:.6f}")

    precisions = [idx / pos for idx, pos in enumerate(positions_meme_classe, 1)] if positions_meme_classe else []
    precision_moyenne = np.mean(precisions) if precisions else 0.0

    print("─"*80)
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Nombre d'images de la même classe trouvées : {len(positions_meme_classe)}")
    print(f"   Positions des images de la classe '{classe_cible}' : {positions_meme_classe}")
    if positions_meme_classe:
        print("   Précision individuelle pour chaque image de la classe :")
        for idx, pos in enumerate(positions_meme_classe, 1):
            print(f"      Image {idx}: {idx} / {pos} = {idx/pos:.4f}")
    print(f"   Précision moyenne (idx/position) : {precision_moyenne:.4f}")
    print("="*80 + "\n")

    # ─── Affichage visuel ───────────────────────────────────────────────────
    cols = min(5, n + 1)           # max 5 colonnes
    rows = (n + 1 + cols - 1) // cols  # nb lignes pour n résultats + image cible

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    # Image cible (1ère case)
    axes[0].imshow(img_cible, cmap=cmap_display)
    axes[0].set_title(f"CIBLE\n{os.path.basename(image_path)}", fontsize=8, fontweight='bold')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)
    axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Images résultats
    for i, (chemin, dist) in enumerate(resultats, 1):
        img_r = cv2.imread(chemin)
        img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_r_display = fn_display(img_r_rgb)
        classe_image = os.path.basename(os.path.dirname(chemin))
        est_meme_classe = classe_image == classe_cible
        couleur_bord = 'green' if est_meme_classe else 'red'

        axes[i].imshow(img_r_display, cmap=cmap_display)
        axes[i].set_title(
            f"#{i} | d={dist:.4f}\n{os.path.basename(chemin)}",
            fontsize=7,
            color='green' if est_meme_classe else 'red'
        )
        for spine in axes[i].spines.values():
            spine.set_edgecolor(couleur_bord)
            spine.set_linewidth(2)
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Masquer les cases vides
    for j in range(len(resultats) + 1, len(axes)):
        axes[j].axis('off')

    legende = [
        mpatches.Patch(color='blue',  label='Image cible'),
        mpatches.Patch(color='green', label='Même classe'),
        mpatches.Patch(color='red',   label='Classe différente'),
    ]
    fig.legend(handles=legende, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Top-{n} images proches | Descripteur: {descripteur.upper()} | Distance: {distance.upper()} | AP={precision_moyenne:.4f}",
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    return resultats, precision_moyenne, fig

