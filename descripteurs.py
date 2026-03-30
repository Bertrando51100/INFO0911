import numpy as np
import cv2
from espaces_couleurs import rgb_vers_hsv, rgb_vers_gris_bt601, rgb_vers_image_indexee


def calculer_histogramme_couleur(image, bins=256, normaliser=False):
    """
    Calcule l'histogramme de couleur d'une image et le retourne sous forme de vecteur.
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    bins : int
        Nombre de bins pour l'histogramme (par défaut 256)
    normaliser : bool
        Si True, normalise l'histogramme (somme = 1)
    
    Retourne:
    ---------
    numpy.ndarray : Vecteur descripteur de taille (bins * nb_canaux,)
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    # Calculer l'histogramme pour chaque canal
    hist_r = np.histogram(image[:,:,0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(image[:,:,1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(image[:,:,2], bins=bins, range=(0, 256))[0]
    
    # Concaténer les 3 histogrammes en un seul vecteur
    vecteur_descripteur = np.concatenate([hist_r, hist_g, hist_b])
    
    # Normaliser si demandé
    if normaliser:
        vecteur_descripteur = vecteur_descripteur.astype(np.float32)
        somme = np.sum(vecteur_descripteur)
        if somme > 0:
            vecteur_descripteur = vecteur_descripteur / somme
    
    return vecteur_descripteur


def calculer_histogramme_gris(image, bins=256, normaliser=False):
    """
    Calcule l'histogramme d'une image en niveaux de gris.
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image en niveaux de gris (H, W) avec valeurs [0-255]
    bins : int
        Nombre de bins pour l'histogramme
    normaliser : bool
        Si True, normalise l'histogramme
    
    Retourne:
    ---------
    numpy.ndarray : Vecteur descripteur de taille (bins,)
    """
    if len(image.shape) == 3:
        # Convertir en niveaux de gris si nécessaire
        image = rgb_vers_gris_bt601(image)
    
    # Calculer l'histogramme
    vecteur_descripteur = np.histogram(image, bins=bins, range=(0, 256))[0]
    
    # Normaliser si demandé
    if normaliser:
        vecteur_descripteur = vecteur_descripteur.astype(np.float32)
        somme = np.sum(vecteur_descripteur)
        if somme > 0:
            vecteur_descripteur = vecteur_descripteur / somme
    
    return vecteur_descripteur


def calculer_histogramme_hsv(image, bins_h=180, bins_s=256, bins_v=256, normaliser=False):
    """
    Calcule l'histogramme HSV d'une image RGB.
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3)
    bins_h : int
        Nombre de bins pour H (Hue) - max 180
    bins_s : int
        Nombre de bins pour S (Saturation)
    bins_v : int
        Nombre de bins pour V (Value)
    normaliser : bool
        Si True, normalise l'histogramme
    
    Retourne:
    ---------
    numpy.ndarray : Vecteur descripteur
    """
    # Convertir en HSV
    hsv = rgb_vers_hsv(image)
    
    # Calculer l'histogramme pour chaque canal
    hist_h = np.histogram(hsv[:,:,0], bins=bins_h, range=(0, 180))[0]
    hist_s = np.histogram(hsv[:,:,1], bins=bins_s, range=(0, 256))[0]
    hist_v = np.histogram(hsv[:,:,2], bins=bins_v, range=(0, 256))[0]
    
    # Concaténer
    vecteur_descripteur = np.concatenate([hist_h, hist_s, hist_v])
    
    # Normaliser
    if normaliser:
        vecteur_descripteur = vecteur_descripteur.astype(np.float32)
        somme = np.sum(vecteur_descripteur)
        if somme > 0:
            vecteur_descripteur = vecteur_descripteur / somme
    
    return vecteur_descripteur

def calculer_histogramme_indexe(image, nb_couleurs=256, normaliser=False):
    """
    Calcule l'histogramme d'une image indexée (palette).
    image : numpy.ndarray (indices ou RGB)
    nb_couleurs : nombre de couleurs de la palette (par défaut 256)
    Retourne : vecteur descripteur de taille (nb_couleurs,)
    """
    # Si image RGB, convertir en indexée
    if len(image.shape) == 3 and image.shape[2] == 3:
        indices, _ = rgb_vers_image_indexee(image, nb_couleurs=nb_couleurs)
    else:
        indices = image
    hist = np.histogram(indices, bins=nb_couleurs, range=(0, nb_couleurs))[0]
    if normaliser:
        hist = hist.astype(np.float32)
        s = np.sum(hist)
        if s > 0:
            hist = hist / s
    return hist


def calculer_histogramme_cumule_gris(image, bins=256, normaliser=False):
    """
    Calcule l'histogramme cumulé d'une image en niveaux de gris.
    Retourne le vecteur cumulé (optionnellement normalisé).
    """
    hist = calculer_histogramme_gris(image, bins=bins, normaliser=normaliser)
    hist_cumule = np.cumsum(hist)
    if normaliser and hist_cumule[-1] > 0:
        hist_cumule = hist_cumule / hist_cumule[-1]
    return hist_cumule

def calculer_histogramme_cumule_couleur(image, bins=256, normaliser=True):
    """
    Calcule l'histogramme cumulé couleur (concaténé R,G,B) d'une image RGB.
    Retourne le vecteur cumulé (optionnellement normalisé).
    """
    hist = calculer_histogramme_couleur(image, bins=bins, normaliser=normaliser)
    # On fait le cumul séparément pour chaque canal
    hist_r_cum = np.cumsum(hist[:bins])
    hist_g_cum = np.cumsum(hist[bins:2*bins])
    hist_b_cum = np.cumsum(hist[2*bins:])
    if normaliser:
        if hist_r_cum[-1] > 0:
            hist_r_cum = hist_r_cum / hist_r_cum[-1]
        if hist_g_cum[-1] > 0:
            hist_g_cum = hist_g_cum / hist_g_cum[-1]
        if hist_b_cum[-1] > 0:
            hist_b_cum = hist_b_cum / hist_b_cum[-1]
    return np.concatenate([hist_r_cum, hist_g_cum, hist_b_cum])

def calculer_histogramme_cumule_indexe(image, nb_couleurs=256, normaliser=False):
    """
    Calcule l'histogramme cumulé d'une image indexée (palette).
    """
    hist = calculer_histogramme_indexe(image, nb_couleurs=nb_couleurs, normaliser=normaliser)
    hist_cumule = np.cumsum(hist)
    if normaliser and hist_cumule[-1] > 0:
        hist_cumule = hist_cumule / hist_cumule[-1]
    return hist_cumule


from skimage.feature import local_binary_pattern
try:
    from skimage.feature import graycomatrix
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix


def _vers_gris(image):
    if len(image.shape) == 3:
        return rgb_vers_gris_bt601(image)
    return image


def _quantifier(image, niveaux=32):
    if niveaux < 2:
        raise ValueError("niveaux doit etre >= 2")
    image_f = np.clip(image.astype(np.float32), 0, 255)
    q = np.floor(image_f * (niveaux / 256.0)).astype(np.int32)
    return np.clip(q, 0, niveaux - 1)


def _iterer_blocs(image, taille_bloc=(32, 32)):
    h, w = image.shape[:2]
    bh, bw = taille_bloc
    if bh <= 0 or bw <= 0:
        raise ValueError("taille_bloc doit contenir des valeurs > 0")

    blocs = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            blocs.append(image[y:y + bh, x:x + bw])
    if not blocs:
        blocs.append(image)
    return blocs

def calculer_lbp_histogramme(image, P=8, R=1, method='uniform'):
    """
    Calcule l'histogramme LBP d'une image NG.
    image : numpy.ndarray (niveaux de gris)
    P : nombre de points voisins
    R : rayon
    method : 'uniform' recommandé
    """
    lbp = local_binary_pattern(image, P, R, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def calculer_entropie(image, bins=256):
    """
    Calcule l'entropie de Shannon comme descripteur d'une image.

    Pour une image couleur (3 canaux), calcule l'entropie de chaque canal
    et concatène les 3 valeurs en un vecteur de taille 3.
    Pour une image en niveaux de gris (2D), retourne un scalaire dans un tableau [1].

    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) ou niveaux de gris (H, W)
    bins : int
        Nombre de bins pour estimer la distribution (défaut 256)

    Retourne:
    ---------
    numpy.ndarray : Vecteur d'entropie (taille 3 pour couleur, 1 pour gris)
    """
    def _entropie_canal(canal):
        hist, _ = np.histogram(canal.ravel(), bins=bins, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist / total
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    if len(image.shape) == 2:
        return np.array([_entropie_canal(image)], dtype=np.float32)
    else:
        return np.array([_entropie_canal(image[:, :, c]) for c in range(image.shape[2])], dtype=np.float32)


def calculer_image_indexee_descripteur(image, nb_couleurs=64):
    """
    Retourne une version vectorisee de l'image indexee.
    Le vecteur est normalise dans [0, 1].
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        indices, _ = rgb_vers_image_indexee(image, nb_couleurs=nb_couleurs)
    else:
        indices = _quantifier(image, niveaux=nb_couleurs)

    denom = float(max(nb_couleurs - 1, 1))
    return (indices.astype(np.float32).ravel() / denom)


def calculer_histogramme_bloc(image, bins=32, taille_bloc=(32, 32), normaliser=True):
    """
    Histogramme calcule par bloc puis concatene en un seul vecteur.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        indices, _ = rgb_vers_image_indexee(image, nb_couleurs=bins)
    else:
        indices = _quantifier(image, niveaux=bins)

    blocs = _iterer_blocs(indices, taille_bloc=taille_bloc)
    features = []
    for bloc in blocs:
        hist, _ = np.histogram(bloc, bins=bins, range=(0, bins))
        hist = hist.astype(np.float32)
        if normaliser:
            s = np.sum(hist)
            if s > 0:
                hist = hist / s
        features.append(hist)
    return np.concatenate(features).astype(np.float32)


def calculer_csv(image, nb_couleurs=64):
    """
    CSV (Color Set Vector): vecteur binaire de presence de couleurs.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        indices, _ = rgb_vers_image_indexee(image, nb_couleurs=nb_couleurs)
    else:
        indices = _quantifier(image, niveaux=nb_couleurs)

    present = np.zeros(nb_couleurs, dtype=np.float32)
    couleurs = np.unique(indices)
    present[couleurs[(couleurs >= 0) & (couleurs < nb_couleurs)]] = 1.0
    return present


def calculer_dcd(image, nb_couleurs=64, top_k=8):
    """
    DCD (Dominant Color Descriptor): indices dominants + proportions.
    Sortie de taille fixe: 2 * top_k.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        indices, _ = rgb_vers_image_indexee(image, nb_couleurs=nb_couleurs)
    else:
        indices = _quantifier(image, niveaux=nb_couleurs)

    hist = np.bincount(indices.ravel(), minlength=nb_couleurs).astype(np.float32)
    total = float(np.sum(hist))
    if total > 0:
        hist = hist / total

    ordre = np.argsort(hist)[::-1][:top_k]
    ids = ordre.astype(np.float32) / float(max(nb_couleurs - 1, 1))
    poids = hist[ordre].astype(np.float32)

    if len(ids) < top_k:
        pad = top_k - len(ids)
        ids = np.pad(ids, (0, pad), mode='constant')
        poids = np.pad(poids, (0, pad), mode='constant')

    return np.concatenate([ids, poids]).astype(np.float32)


def calculer_ccd(image, nb_couleurs=32, seuil_coherence=30):
    """
    CCD (Color Coherence Descriptor): pixels coherents/incoherents par couleur.
    Sortie de taille fixe: 2 * nb_couleurs.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        indices, _ = rgb_vers_image_indexee(image, nb_couleurs=nb_couleurs)
    else:
        indices = _quantifier(image, niveaux=nb_couleurs)

    coherents = np.zeros(nb_couleurs, dtype=np.float32)
    incoherents = np.zeros(nb_couleurs, dtype=np.float32)

    for c in range(nb_couleurs):
        masque = (indices == c).astype(np.uint8)
        if np.count_nonzero(masque) == 0:
            continue

        nb_labels, labels = cv2.connectedComponents(masque)
        for label in range(1, nb_labels):
            taille = int(np.count_nonzero(labels == label))
            if taille >= seuil_coherence:
                coherents[c] += taille
            else:
                incoherents[c] += taille

    total = float(indices.size)
    if total > 0:
        coherents /= total
        incoherents /= total

    return np.concatenate([coherents, incoherents]).astype(np.float32)


def calculer_matrice_concurrence(image, niveaux=16, distances=(1,), angles=(0,), normaliser=True):
    """
    Matrice de cooccurrence (GLCM) aplatit en vecteur.
    """
    gris = _vers_gris(image)
    q = _quantifier(gris, niveaux=niveaux).astype(np.uint8)
    glcm = graycomatrix(
        q,
        distances=list(distances),
        angles=list(angles),
        levels=niveaux,
        symmetric=True,
        normed=normaliser,
    )
    return glcm.astype(np.float32).ravel()


def calculer_lbp(image, P=8, R=1, method='uniform'):
    """
    LBP pixel-a-pixel, retourne une image LBP vectorisee normalisee.
    """
    gris = _vers_gris(image)
    lbp = local_binary_pattern(gris, P, R, method)
    max_lbp = float(max(np.max(lbp), 1.0))
    return (lbp.astype(np.float32).ravel() / max_lbp)


def calculer_histogramme_bloc_lbp(image, P=8, R=1, method='uniform', taille_bloc=(32, 32)):
    """
    Histogrammes LBP par bloc concatenes.
    """
    gris = _vers_gris(image)
    blocs = _iterer_blocs(gris, taille_bloc=taille_bloc)

    if method == 'uniform':
        n_bins = P + 2
    else:
        n_bins = int(P * (P - 1) + 3)

    features = []
    for bloc in blocs:
        lbp = local_binary_pattern(bloc, P, R, method)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.append(hist.astype(np.float32))

    return np.concatenate(features).astype(np.float32)


def _gradients(image):
    gris = _vers_gris(image).astype(np.float32)
    gy, gx = np.gradient(gris)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    return magnitude, orientation


def calculer_histogramme_direction_gradient(image, bins=9, normaliser=True):
    """
    Histogramme de direction du gradient (non pondere).
    """
    _, orientation = _gradients(image)
    hist, _ = np.histogram(orientation.ravel(), bins=bins, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    if normaliser:
        s = np.sum(hist)
        if s > 0:
            hist = hist / s
    return hist


def calculer_histogramme_pondere_par_norme(image, bins=9, normaliser=True):
    """
    Histogramme de direction du gradient pondere par la norme.
    """
    magnitude, orientation = _gradients(image)
    hist, _ = np.histogram(
        orientation.ravel(),
        bins=bins,
        range=(0.0, 180.0),
        weights=magnitude.ravel(),
    )
    hist = hist.astype(np.float32)
    if normaliser:
        s = np.sum(hist)
        if s > 0:
            hist = hist / s
    return hist


def calculer_histogramme_bloc_direction_gradient(image, bins=9, taille_bloc=(16, 16), normaliser=True):
    """
    Histogramme de direction du gradient par bloc (type HOG simple).
    """
    magnitude, orientation = _gradients(image)
    h, w = orientation.shape
    bh, bw = taille_bloc
    features = []

    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            bloc_o = orientation[y:y + bh, x:x + bw]
            bloc_m = magnitude[y:y + bh, x:x + bw]
            hist, _ = np.histogram(
                bloc_o.ravel(),
                bins=bins,
                range=(0.0, 180.0),
                weights=bloc_m.ravel(),
            )
            hist = hist.astype(np.float32)
            if normaliser:
                s = np.sum(hist)
                if s > 0:
                    hist = hist / s
            features.append(hist)

    if not features:
        hist, _ = np.histogram(
            orientation.ravel(),
            bins=bins,
            range=(0.0, 180.0),
            weights=magnitude.ravel(),
        )
        hist = hist.astype(np.float32)
        if normaliser:
            s = np.sum(hist)
            if s > 0:
                hist = hist / s
        return hist

    return np.concatenate(features).astype(np.float32)