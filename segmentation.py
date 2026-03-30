"""
Module de segmentation d'images
Contient les fonctions pour la binarisation et la segmentation par kmeans
Basé sur les notebooks binarisation_images.ipynb et segmentation_kmeans.ipynb
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _palette_rgb(n_classes=256, seed=42):
    """Crée une palette pseudo-aléatoire reproductible pour colorer les masques."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(n_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def coloriser_masque(mask):
    """Convertit un masque 2D (labels) en image RGB colorée."""
    mask_u8 = mask.astype(np.uint8)
    palette = _palette_rgb(256)
    return palette[mask_u8]


# ============================================================================
# BINARISATION - Basée sur binarisation_images.ipynb
# ============================================================================

# ---- MÉTHODES GLOBALES SUR NIVEAUX DE GRIS ----

def binarisation_niveaux_gris(image, seuil=None, methode='otsu'):
    """
    Binarise une image en utilisant les niveaux de gris (Otsu ou manuel).
    
    Parameters:
    -----------
    image : array RGB (H, W, 3)
    seuil : int, optional (0-255)
        Seuil manuel. Si None, utilise Otsu.
    methode : str, 'otsu' ou 'manual'
    
    Returns:
    --------
    gray, binary, hist, seuil_utilise
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    if methode == 'otsu' or seuil is None:
        seuil_utilise, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        seuil_utilise, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    
    return gray, binary, hist, int(seuil_utilise)


def binarisation_gris_moyenne(image):
    """Binarisation niveaux de gris avec seuil = moyenne"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seuil = int(np.mean(gray))
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    return gray, binary, seuil


def binarisation_gris_mediane(image):
    """Binarisation niveaux de gris avec seuil = médiane"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seuil = int(np.median(gray))
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    return gray, binary, seuil


def binarisation_gris_minmax(image):
    """Binarisation niveaux de gris avec seuil = (min + max) / 2"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seuil = int((np.min(gray) + np.max(gray)) / 2)
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    return gray, binary, seuil


def binarisation_gris_ecart_type(image):
    """Binarisation niveaux de gris avec seuil = écart-type"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seuil = int(np.std(gray))
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    return gray, binary, seuil


def binarisation_gris_ptile(image, percentile=50):
    """Binarisation niveaux de gris avec seuil = P-tile (percentile)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seuil = int(np.percentile(gray, percentile))
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    return gray, binary, seuil


def binarisation_gris_moyenne_tronquee(image, pourcentage=10):
    """Binarisation niveaux de gris avec seuil = moyenne tronquée"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_flat = gray.flatten()
    percentile_bas = np.percentile(gray_flat, pourcentage)
    percentile_haut = np.percentile(gray_flat, 100 - pourcentage)
    gray_tronque = gray_flat[(gray_flat >= percentile_bas) & (gray_flat <= percentile_haut)]
    seuil = int(np.mean(gray_tronque))
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)
    return gray, binary, seuil


# ---- MÉTHODES GLOBALES SUR TEINTE (HUE) ----

def binarisation_teinte(image, seuil_bas=None, seuil_haut=None):
    """
    Binarise une image en utilisant la composante H (teinte) de HSV.
    
    Parameters:
    -----------
    image : array RGB
    seuil_bas, seuil_haut : int, optional (0-180 pour OpenCV)
        Plage de teintes à conserver. Si None, utilise Otsu.
    
    Returns:
    --------
    hsv, h_channel, binary, hist
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
    
    if seuil_bas is None or seuil_haut is None:
        seuil_utilise, binary = cv2.threshold(h_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.inRange(h_channel, seuil_bas, seuil_haut)
    
    return hsv, h_channel, binary, hist


def binarisation_hue_moyenne(image):
    """Binarisation sur Hue avec seuil = moyenne"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    seuil = int(np.mean(h_channel))
    _, binary = cv2.threshold(h_channel, seuil, 255, cv2.THRESH_BINARY)
    return h_channel, binary, seuil


def binarisation_hue_mediane(image):
    """Binarisation sur Hue avec seuil = médiane"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    seuil = int(np.median(h_channel))
    _, binary = cv2.threshold(h_channel, seuil, 255, cv2.THRESH_BINARY)
    return h_channel, binary, seuil


def binarisation_hue_minmax(image):
    """Binarisation sur Hue avec seuil = (min + max) / 2"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    seuil = int((np.min(h_channel) + np.max(h_channel)) / 2)
    _, binary = cv2.threshold(h_channel, seuil, 255, cv2.THRESH_BINARY)
    return h_channel, binary, seuil


def binarisation_hue_ecart_type(image):
    """Binarisation sur Hue avec seuil = écart-type"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    seuil = int(np.std(h_channel))
    _, binary = cv2.threshold(h_channel, seuil, 255, cv2.THRESH_BINARY)
    return h_channel, binary, seuil


def binarisation_hue_ptile(image, percentile=50):
    """Binarisation sur Hue avec seuil = P-tile (percentile)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    seuil = int(np.percentile(h_channel, percentile))
    _, binary = cv2.threshold(h_channel, seuil, 255, cv2.THRESH_BINARY)
    return h_channel, binary, seuil


def binarisation_hue_moyenne_tronquee(image, pourcentage=10):
    """Binarisation sur Hue avec seuil = moyenne tronquée"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    h_flat = h_channel.flatten()
    percentile_bas = np.percentile(h_flat, pourcentage)
    percentile_haut = np.percentile(h_flat, 100 - pourcentage)
    h_tronque = h_flat[(h_flat >= percentile_bas) & (h_flat <= percentile_haut)]
    seuil = int(np.mean(h_tronque))
    _, binary = cv2.threshold(h_channel, seuil, 255, cv2.THRESH_BINARY)
    return h_channel, binary, seuil


# ---- MÉTHODE ISODATA ----

def binarisation_isodata_moyenne(image, max_iterations=100, tolerance=0.5):
    """
    Binarisation avec la méthode ISODATA initialisée avec la MOYENNE.
    Converge vers un seuil optimal en itérant.
    
    Parameters:
    -----------
    image : array RGB
    max_iterations : int
    tolerance : float, seuil de convergence
    
    Returns:
    --------
    gray, binary, seuil_final, iterations, seuil_initial
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    seuil_initial = np.mean(gray)
    seuil = seuil_initial
    
    for i in range(max_iterations):
        classe1 = gray[gray <= seuil]
        classe2 = gray[gray > seuil]
        
        moyenne1 = np.mean(classe1) if len(classe1) > 0 else 0
        moyenne2 = np.mean(classe2) if len(classe2) > 0 else 255
        
        nouveau_seuil = (moyenne1 + moyenne2) / 2
        
        if abs(nouveau_seuil - seuil) < tolerance:
            seuil = nouveau_seuil
            break
        
        seuil = nouveau_seuil
    
    seuil_final = int(seuil)
    _, binary = cv2.threshold(gray, seuil_final, 255, cv2.THRESH_BINARY)
    
    return gray, binary, seuil_final, i + 1, int(seuil_initial)


def binarisation_isodata_4coins(image, max_iterations=100, tolerance=0.5, coin_size=10):
    """
    Binarisation avec la méthode ISODATA initialisée avec MOYENNE DES 4 COINS.
    Plus robuste pour images avec éclairage non uniforme.
    
    Parameters:
    -----------
    image : array RGB
    max_iterations : int
    tolerance : float
    coin_size : int, taille des coins en pixels
    
    Returns:
    --------
    gray, binary, seuil_final, iterations, seuil_initial
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    coin_haut_gauche = gray[0:coin_size, 0:coin_size]
    coin_haut_droit = gray[0:coin_size, w-coin_size:w]
    coin_bas_gauche = gray[h-coin_size:h, 0:coin_size]
    coin_bas_droit = gray[h-coin_size:h, w-coin_size:w]
    
    seuil_initial = (np.mean(coin_haut_gauche) + np.mean(coin_haut_droit) +
                     np.mean(coin_bas_gauche) + np.mean(coin_bas_droit)) / 4
    
    seuil = seuil_initial
    
    for i in range(max_iterations):
        classe1 = gray[gray <= seuil]
        classe2 = gray[gray > seuil]
        
        moyenne1 = np.mean(classe1) if len(classe1) > 0 else 0
        moyenne2 = np.mean(classe2) if len(classe2) > 0 else 255
        
        nouveau_seuil = (moyenne1 + moyenne2) / 2
        
        if abs(nouveau_seuil - seuil) < tolerance:
            seuil = nouveau_seuil
            break
        
        seuil = nouveau_seuil
    
    seuil_final = int(seuil)
    _, binary = cv2.threshold(gray, seuil_final, 255, cv2.THRESH_BINARY)
    
    return gray, binary, seuil_final, i + 1, int(seuil_initial)


# ---- MÉTHODES LOCALES ----

def binarisation_locale_moyenne(image, taille_fenetre=7):
    """
    Binarisation locale avec seuil = moyenne dans une fenêtre.
    Adaptatif à l'éclairage local.
    
    Parameters:
    -----------
    image : array RGB
    taille_fenetre : int, taille de la fenêtre
    
    Returns:
    --------
    gray, binary
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_local = cv2.blur(gray, (taille_fenetre, taille_fenetre))
    binary = np.where(gray > mean_local, 255, 0).astype(np.uint8)
    return gray, binary


def binarisation_locale_mediane(image, taille_fenetre=7):
    """
    Binarisation locale avec seuil = médiane dans une fenêtre.
    Robuste au bruit.
    
    Parameters:
    -----------
    image : array RGB
    taille_fenetre : int, taille de la fenêtre
    
    Returns:
    --------
    gray, binary
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    median_local = cv2.medianBlur(gray, taille_fenetre)
    binary = np.where(gray > median_local, 255, 0).astype(np.uint8)
    return gray, binary


def binarisation_locale_minmax(image, taille_fenetre=7):
    """
    Binarisation locale avec seuil = (min + max) / 2 dans une fenêtre.
    Peut être coûteux en calcul. Utile pour texte avec ombres.
    
    Parameters:
    -----------
    image : array RGB
    taille_fenetre : int, taille de la fenêtre
    
    Returns:
    --------
    gray, binary
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pad = taille_fenetre // 2
    gray_padded = np.pad(gray, pad, mode='reflect')
    binary = np.zeros_like(gray)
    h, w = gray.shape
    
    for i in range(h):
        for j in range(w):
            fenetre = gray_padded[i:i+taille_fenetre, j:j+taille_fenetre]
            seuil_local = (np.min(fenetre) + np.max(fenetre)) / 2
            binary[i, j] = 255 if gray[i, j] > seuil_local else 0
    
    return gray, binary


# ---- MÉTHODE ADAPTATIVE (OPENCV) ----

def binarisation_adaptatif(image, block_size=11):
    """
    Binarisation adaptative locale (par bloc) - OpenCV optimisée.
    Plus rapide que binarisation_locale_*.
    
    Parameters:
    -----------
    image : array RGB
    block_size : int, taille du bloc (doit être impair)
    
    Returns:
    --------
    gray, binary
    """
    if block_size % 2 == 0:
        block_size += 1
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, 2
    )
    
    return gray, binary


# ============================================================================
# ALGORITHMES DEMANDÉS: UPSAMPLING / FCN / U-NET / SEGNET / PSPNET
# ============================================================================

def up_sampling(image_rgb, facteur=2, methode='bilinear'):
    """
    Up sampling d'image par redimensionnement.

    Parameters:
    -----------
    image_rgb : array RGB
    facteur : float
        Facteur d'agrandissement (>1)
    methode : str
        'nearest', 'bilinear' ou 'bicubic'

    Returns:
    --------
    image_up : image agrandie
    """
    if facteur <= 1:
        facteur = 2

    h, w = image_rgb.shape[:2]
    new_size = (int(w * facteur), int(h * facteur))

    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
    }
    inter = interpolation_map.get(methode.lower(), cv2.INTER_LINEAR)
    image_up = cv2.resize(image_rgb, new_size, interpolation=inter)
    return image_up


def segmentation_fcn(image_rgb):
    """
    Segmentation avec FCN (torchvision pretrained si disponible).
    Fallback: K-Means si torch/torchvision indisponible.

    Returns:
    --------
    mask : masque labels 2D
    image_colorisee : visualisation RGB du masque
    info : dict
    """
    try:
        import torch
        import torchvision
        from PIL import Image

        weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
        model.eval()

        preprocess = weights.transforms()
        input_tensor = preprocess(Image.fromarray(image_rgb)).unsqueeze(0)

        with torch.no_grad():
            out = model(input_tensor)['out'][0]

        mask = out.argmax(0).cpu().numpy().astype(np.uint8)
        image_colorisee = coloriser_masque(mask)

        categories = weights.meta.get('categories', [])
        info = {
            'mode': 'fcn_pretrained',
            'classes_detectees': int(np.unique(mask).size),
            'nb_classes_modele': int(len(categories)) if categories else None,
        }
        return mask, image_colorisee, info

    except Exception:
        seg_img, labels, centres, inertia = segmentation_kmeans(
            image_rgb, k=4, espace_couleur='rgb', random_state=42
        )
        mask = labels.reshape(image_rgb.shape[:2]).astype(np.uint8)
        image_colorisee = coloriser_masque(mask)
        info = {
            'mode': 'fallback_kmeans',
            'classes_detectees': int(np.unique(mask).size),
            'inertia': float(inertia),
        }
        return mask, image_colorisee, info


def segmentation_unet(image_rgb):
    """
    Approximation de segmentation type U-Net (fallback pédagogique).
    Utilise une combinaison de seuillage adaptatif + morphologie.

    Returns:
    --------
    mask : masque binaire 2D
    image_colorisee : visualisation RGB
    info : dict
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = (mask > 0).astype(np.uint8)
    image_colorisee = coloriser_masque(mask)
    info = {
        'mode': 'unet_style_fallback',
        'classes_detectees': int(np.unique(mask).size),
    }
    return mask, image_colorisee, info


def segmentation_segnet(image_rgb):
    """
    Approximation SegNet via watershed (fallback robuste sans modèle).

    Returns:
    --------
    mask : labels 2D
    image_colorisee : visualisation RGB
    info : dict
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)
    markers[markers < 0] = 0
    mask = markers.astype(np.uint8)

    image_colorisee = coloriser_masque(mask)
    info = {
        'mode': 'segnet_style_fallback_watershed',
        'classes_detectees': int(np.unique(mask).size),
    }
    return mask, image_colorisee, info


def segmentation_pspnet(image_rgb):
    """
    Approximation PSPNet via contexte multi-échelle + K-Means.

    Returns:
    --------
    mask : labels 2D
    image_colorisee : visualisation RGB
    info : dict
    """
    img = image_rgb.astype(np.float32)
    h, w = img.shape[:2]

    # Contexte multi-échelle (idée type pyramid pooling)
    s1 = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    s1 = cv2.resize(s1, (w, h), interpolation=cv2.INTER_LINEAR)
    s2 = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    s2 = cv2.resize(s2, (w, h), interpolation=cv2.INTER_LINEAR)

    feat = np.concatenate([
        img,
        s1,
        s2,
    ], axis=2)

    X = feat.reshape(-1, feat.shape[2]).astype(np.float32)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    mask = labels.reshape(h, w).astype(np.uint8)
    image_colorisee = coloriser_masque(mask)
    info = {
        'mode': 'pspnet_style_fallback_multiscale_kmeans',
        'classes_detectees': int(np.unique(mask).size),
        'inertia': float(kmeans.inertia_),
    }
    return mask, image_colorisee, info


# ============================================================================
# SEGMENTATION K-MEANS - Basée sur segmentation_kmeans.ipynb
# ============================================================================

def pretraiter_image(image_rgb, target_size=(200, 200)):
    """
    Prétraite une image pour la segmentation kmeans.
    
    Parameters:
    -----------
    image_rgb : array RGB (H, W, 3)
    target_size : tuple
        Taille de redimensionnement (hauteur, largeur)
    
    Returns:
    --------
    image_redim : image redimensionnée en RGB
    image_gris : image en niveaux de gris
    """
    # Redimensionner pour cohérence
    image_redim = cv2.resize(image_rgb, target_size[::-1])
    
    # Convertir en niveaux de gris pour comparaison
    image_gris = cv2.cvtColor(image_redim, cv2.COLOR_RGB2GRAY)
    
    return image_redim, image_gris


def segmentation_kmeans(image_rgb, k=3, espace_couleur='rgb', random_state=42):
    """
    Segmente une image par K-Means clustering.
    
    Parameters:
    -----------
    image_rgb : array RGB (H, W, 3)
    k : int
        Nombre de clusters
    espace_couleur : str
        'rgb', 'hsv', ou 'gris'
    random_state : int
        Pour reproductibilité
    
    Returns:
    --------
    image_segmentee : image avec pixels remplacés par centres de clusters
    labels : array de labels (cluster pour chaque pixel)
    centres : centres des clusters
    inertia : inertie (somme des distances)
    """
    h, w = image_rgb.shape[:2]
    
    # Préparer l'espace de couleur
    if espace_couleur == 'hsv':
        img_work = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    elif espace_couleur == 'gris':
        img_work = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        img_work = np.expand_dims(img_work, axis=2)  # (H, W) -> (H, W, 1)
    else:  # rgb
        img_work = image_rgb
    
    # Convertir en matrice (pixels x features)
    X = img_work.reshape(-1, img_work.shape[2]).astype(np.float32)
    
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    centres = kmeans.cluster_centers_.astype(np.uint8)
    inertia = kmeans.inertia_
    
    # Reconstruire l'image segmentée
    image_segmentee = centres[labels].reshape(h, w, img_work.shape[2])
    
    # Reconvertir si nécessaire
    if espace_couleur == 'hsv':
        image_segmentee = cv2.cvtColor(image_segmentee, cv2.COLOR_HSV2RGB)
    elif espace_couleur == 'gris':
        image_segmentee = np.squeeze(image_segmentee, axis=2)
    
    return image_segmentee, labels, centres, inertia


# ============================================================================
# FONCTION UTILITAIRE POUR CALCULER STATISTIQUES
# ============================================================================

def calculer_statistiques_binarisation(image_gris, image_binaire, seuil):
    """
    Calcule les statistiques pour une binarisation.
    
    Parameters:
    -----------
    image_gris : array niveaux de gris
    image_binaire : array binaire (0-255)
    seuil : int
    
    Returns:
    --------
    dict : dictionnaire contenant les statistiques
    """
    stats = {
        'seuil': int(seuil),
        'pixels_blancs': int(np.sum(image_binaire == 255)),
        'pixels_blancs_pct': float(np.sum(image_binaire == 255) / image_binaire.size * 100),
        'pixels_noirs': int(np.sum(image_binaire == 0)),
        'pixels_noirs_pct': float(np.sum(image_binaire == 0) / image_binaire.size * 100),
        'gris_min': int(np.min(image_gris)),
        'gris_max': int(np.max(image_gris)),
        'gris_mean': float(np.mean(image_gris)),
        'gris_std': float(np.std(image_gris)),
    }
    return stats
