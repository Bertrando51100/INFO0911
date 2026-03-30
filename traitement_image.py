## Fichier qui comporte toute les fonctions de traitement d'images 
import numpy as np


## Application de filtres et padding

# Création d'un filtre aléatoire
def creation_filtre_aleatoire(taille, min_val=-1.0, max_val=1.0):
    """
    Crée un filtre carré aléatoire de taille (taille x taille).
    Les valeurs sont tirées uniformément entre min_val et max_val.
    """
    return np.random.uniform(min_val, max_val, (taille, taille)).astype(np.float32)

# Filtre de prewitt
def filtre_prewitt():
    """Retourne les filtres Prewitt pour les directions x et y."""
    fx = np.array([[ -1, 0, 1],
                   [ -1, 0, 1],
                   [ -1, 0, 1]], dtype=np.float32)
    fy = np.array([[ 1,  1,  1],
                   [ 0,  0,  0],
                   [-1, -1, -1]], dtype=np.float32)
    return fx, fy

# Filtre de sobel
def filtre_sobel():
    """Retourne les filtres Sobel pour les directions x et y."""
    fx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]], dtype=np.float32)
    fy = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
    return fx, fy

# Filtre de roberts
def filtre_roberts():
    """Retourne les filtres Roberts pour les directions x et y."""
    fx = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)
    fy = np.array([[0, 1],
                   [-1, 0]], dtype=np.float32)
    return fx, fy

# Filtre laplacien
def filtre_laplacien():
    """Retourne le filtre Laplacien classique 3x3."""
    f = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)
    return f

# Filtre scharr
def filtre_scharr():
    """Retourne les filtres Scharr pour les directions x et y."""
    fx = np.array([[ -3, 0, 3],
                   [-10, 0,10],
                   [ -3, 0, 3]], dtype=np.float32)
    fy = np.array([[ 3, 10,  3],
                   [ 0,  0,  0],
                   [-3,-10, -3]], dtype=np.float32)
    return fx, fy

# Ajout de padding à une image
def ajouter_padding(image, pad_y, pad_x, mode='constant', valeur=0):
    """
    Ajoute du padding à une image (2D ou 3D).
    pad_y : nombre de lignes à ajouter en haut et en bas
    pad_x : nombre de colonnes à ajouter à gauche et à droite
    mode : 'constant', 'edge', etc. (voir np.pad)
    valeur : valeur de remplissage si mode='constant'
    """
    if len(image.shape) == 2:
        pad_width = ((pad_y, pad_y), (pad_x, pad_x))
    elif len(image.shape) == 3:
        pad_width = ((pad_y, pad_y), (pad_x, pad_x), (0, 0))
    else:
        raise ValueError("Image doit être 2D ou 3D")
    if mode == 'constant':
        return np.pad(image, pad_width, mode=mode, constant_values=valeur)
    return np.pad(image, pad_width, mode=mode)


## Application de la convolution et cross-correlation

# Application de la cross -corrélation avec padding
def cross_correlation(image, filtre, stride=1):
    """
    Applique une cross-corrélation entre l'image et le filtre avec un stride donné.
    image : 2D numpy array (niveaux de gris)
    filtre : 2D numpy array (filtre)
    stride : int (pas du déplacement)
    Retourne l'image filtrée.
    """
    h, w = image.shape
    fh, fw = filtre.shape
    pad_y = fh // 2
    pad_x = fw // 2
    image_pad = ajouter_padding(image, pad_y, pad_x)
    out_h = (h - fh) // stride + 1 if stride > 1 else h
    out_w = (w - fw) // stride + 1 if stride > 1 else w
    resultat = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(0, out_h):
        for j in range(0, out_w):
            y = i * stride
            x = j * stride
            region = image_pad[y:y+fh, x:x+fw]
            resultat[i, j] = np.sum(region * filtre)
    return resultat

# Application de la convolution avec padding
def convolution(image, filtre, stride=1):
    """
    Applique une convolution entre l'image et le filtre avec un stride donné.
    image : 2D numpy array (niveaux de gris)
    filtre : 2D numpy array (filtre)
    stride : int (pas du déplacement)
    Retourne l'image convoluée.
    """
    # Retourner le filtre retourné (180°) pour la convolution
    filtre_conv = np.flipud(np.fliplr(filtre))
    return cross_correlation(image, filtre_conv, stride=stride)


## Ajout de bruits

# Ajout de bruit gaussien
def ajouter_bruit_gaussien(image, mean=0, sigma=20):
    """Ajoute un bruit gaussien à une image (uint8)."""
    bruit = np.random.normal(mean, sigma, image.shape)
    image_bruitee = image.astype(np.float32) + bruit
    return np.clip(image_bruitee, 0, 255).astype(np.uint8)

# Ajout de bruit poivre et sel
def ajouter_bruit_poivre_sel(image, proportion=0.05):
    """Ajoute un bruit poivre et sel à une image (uint8)."""
    img_bruitee = image.copy()
    nb_pixels = image.size
    nb_poivre = int(proportion * nb_pixels / 2)
    nb_sel = int(proportion * nb_pixels / 2)

    # Poivre (noir)
    coords = [np.random.randint(0, i - 1, nb_poivre) for i in image.shape]
    img_bruitee[tuple(coords)] = 0

    # Sel (blanc)
    coords = [np.random.randint(0, i - 1, nb_sel) for i in image.shape]
    img_bruitee[tuple(coords)] = 255

    return img_bruitee

# Ajout de bruit de Poisson
def ajouter_bruit_poisson(image):
    """Ajoute un bruit de Poisson à une image (uint8)."""
    # Poisson attend des valeurs positives, donc on normalise si besoin
    image_float = image.astype(np.float32)
    # Génère le bruit de Poisson
    bruit = np.random.poisson(image_float).astype(np.float32)
    return np.clip(bruit, 0, 255).astype(np.uint8)