import numpy as np
from PIL import Image


# =============================================================================
# CONVERSIONS RGB VERS AUTRES ESPACES DE COULEURS
# =============================================================================

def rgb_vers_gris_moyenne(image):
    """
    Convertit RGB en niveaux de gris avec la méthode moyenne simple.
    Formule : Gray = (R + G + B) / 3
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image en niveaux de gris [0-255] uint8
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    R = image[:,:,0].astype(np.float32)
    G = image[:,:,1].astype(np.float32)
    B = image[:,:,2].astype(np.float32)
    
    gray = (R + G + B) / 3.0
    
    return np.clip(gray, 0, 255).astype(np.uint8)


def rgb_vers_gris_bt601(image):
    """
    Convertit RGB en niveaux de gris avec la méthode ITU-R BT.601.
    Formule : Gray = 0.299*R + 0.587*G + 0.114*B
    
    Standard TV analogique (NTSC, PAL, SECAM).
    Utilisé par défaut dans OpenCV.
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image en niveaux de gris [0-255] uint8
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    R = image[:,:,0].astype(np.float32)
    G = image[:,:,1].astype(np.float32)
    B = image[:,:,2].astype(np.float32)
    
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    return np.clip(gray, 0, 255).astype(np.uint8)


def rgb_vers_gris_bt709(image):
    """
    Convertit RGB en niveaux de gris avec la méthode ITU-R BT.709.
    Formule : Gray = 0.2126*R + 0.7152*G + 0.0722*B
    
    Standard HDTV (télévision haute définition).
    Plus précis pour les écrans modernes.
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image en niveaux de gris [0-255] uint8
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    R = image[:,:,0].astype(np.float32)
    G = image[:,:,1].astype(np.float32)
    B = image[:,:,2].astype(np.float32)
    
    gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    return np.clip(gray, 0, 255).astype(np.uint8)


def rgb_vers_hsv(image):
    """
    Convertit RGB en HSV (Hue, Saturation, Value).
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image HSV avec H∈[0,179], S∈[0,255], V∈[0,255]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    img_norm = image.astype(np.float32) / 255.0
    R, G, B = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
    
    V = np.max(img_norm, axis=2)
    m = np.min(img_norm, axis=2)
    delta = V - m
    
    # Saturation
    S = np.zeros_like(V)
    mask_v = V != 0
    S[mask_v] = delta[mask_v] / V[mask_v]
    
    # Teinte
    H = np.zeros_like(V)
    mask = delta != 0
    
    idx = (V == R) & mask
    H[idx] = 60 * ((G[idx] - B[idx]) / delta[idx])
    
    idx = (V == G) & mask
    H[idx] = 60 * (2 + (B[idx] - R[idx]) / delta[idx])
    
    idx = (V == B) & mask
    H[idx] = 60 * (4 + (R[idx] - G[idx]) / delta[idx])
    
    H[H < 0] += 360
    
    # Format OpenCV : H/2, S*255, V*255
    return np.stack([H / 2, S * 255, V * 255], axis=2).astype(np.uint8)


def rgb_vers_hls(image):
    """
    Convertit RGB en HLS (Hue, Lightness, Saturation).
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image HLS avec H∈[0,179], L∈[0,255], S∈[0,255]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    img_norm = image.astype(np.float32) / 255.0
    R, G, B = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
    
    max_val = np.max(img_norm, axis=2)
    min_val = np.min(img_norm, axis=2)
    delta = max_val - min_val
    
    # Luminosité
    L = (max_val + min_val) / 2
    
    # Saturation
    S = np.zeros_like(L)
    mask = delta != 0
    
    mask_l_low = mask & (L < 0.5)
    S[mask_l_low] = delta[mask_l_low] / (max_val[mask_l_low] + min_val[mask_l_low])
    
    mask_l_high = mask & (L >= 0.5)
    S[mask_l_high] = delta[mask_l_high] / (2.0 - max_val[mask_l_high] - min_val[mask_l_high])
    
    # Teinte
    H = np.zeros_like(L)
    
    idx = (max_val == R) & mask
    H[idx] = 60 * ((G[idx] - B[idx]) / delta[idx])
    
    idx = (max_val == G) & mask
    H[idx] = 60 * (2 + (B[idx] - R[idx]) / delta[idx])
    
    idx = (max_val == B) & mask
    H[idx] = 60 * (4 + (R[idx] - G[idx]) / delta[idx])
    
    H[H < 0] += 360
    
    return np.stack([H / 2, L * 255, S * 255], axis=2).astype(np.uint8)


def rgb_vers_ycrcb(image):
    """
    Convertit RGB en YCrCb (Luminance + Chrominance).
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image YCrCb avec Y, Cr, Cb ∈ [0,255]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    R = image[:,:,0].astype(np.float32)
    G = image[:,:,1].astype(np.float32)
    B = image[:,:,2].astype(np.float32)
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128
    
    return np.stack([Y, Cr, Cb], axis=2).clip(0, 255).astype(np.uint8)


def rgb_vers_lab(image):
    """
    Convertit RGB en Lab (via XYZ).
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image Lab avec L, a, b ∈ [0,255]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    # RGB -> XYZ
    img_norm = image.astype(np.float32) / 255.0
    
    # Correction gamma (sRGB)
    mask = img_norm > 0.04045
    img_norm[mask] = np.power((img_norm[mask] + 0.055) / 1.055, 2.4)
    img_norm[~mask] = img_norm[~mask] / 12.92
    
    # Matrice de transformation RGB -> XYZ (D65)
    R, G, B = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
    X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
    Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
    Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041
    
    # XYZ -> Lab
    X = X / 0.95047  # D65
    Y = Y / 1.00000
    Z = Z / 1.08883
    
    def f(t):
        delta = 6/29
        mask = t > delta**3
        result = np.zeros_like(t)
        result[mask] = np.power(t[mask], 1/3)
        result[~mask] = t[~mask] / (3 * delta**2) + 4/29
        return result
    
    fX = f(X)
    fY = f(Y)
    fZ = f(Z)
    
    L = 116 * fY - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)
    
    # Normalisation pour OpenCV [0, 255]
    L = L * 255 / 100
    a = a + 128
    b = b + 128
    
    return np.stack([L, a, b], axis=2).clip(0, 255).astype(np.uint8)


def rgb_vers_luv(image):
    """
    Convertit RGB en Luv (via XYZ).
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image Luv avec L, u, v ∈ [0,255]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    # RGB -> XYZ (même processus que Lab)
    img_norm = image.astype(np.float32) / 255.0
    
    mask = img_norm > 0.04045
    img_norm[mask] = np.power((img_norm[mask] + 0.055) / 1.055, 2.4)
    img_norm[~mask] = img_norm[~mask] / 12.92
    
    R, G, B = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
    X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
    Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
    Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041
    
    # XYZ -> Luv
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883  # D65
    
    epsilon = 0.008856
    kappa = 903.3
    
    yr = Y / Yn
    
    L = np.zeros_like(Y)
    mask = yr > epsilon
    L[mask] = 116 * np.power(yr[mask], 1/3) - 16
    L[~mask] = kappa * yr[~mask]
    
    denom = X + 15*Y + 3*Z
    denom[denom == 0] = 1e-10
    
    u_prime = 4*X / denom
    v_prime = 9*Y / denom
    
    u_prime_n = 4*Xn / (Xn + 15*Yn + 3*Zn)
    v_prime_n = 9*Yn / (Xn + 15*Yn + 3*Zn)
    
    u = 13 * L * (u_prime - u_prime_n)
    v = 13 * L * (v_prime - v_prime_n)
    
    # Normalisation OpenCV
    L = L * 255 / 100
    u = (u + 134) * 255 / 354
    v = (v + 140) * 255 / 262
    
    return np.stack([L, u, v], axis=2).clip(0, 255).astype(np.uint8)


def rgb_vers_xyz(image):
    """
    Convertit RGB en XYZ.
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image XYZ ∈ [0,255]
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    img_norm = image.astype(np.float32) / 255.0
    
    # Correction gamma
    mask = img_norm > 0.04045
    img_norm[mask] = np.power((img_norm[mask] + 0.055) / 1.055, 2.4)
    img_norm[~mask] = img_norm[~mask] / 12.92
    
    R, G, B = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
    
    X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
    Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
    Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041
    
    # Normalisation pour OpenCV
    return (np.stack([X, Y, Z], axis=2).clip(0, 1) * 255).astype(np.uint8)


def rgb_vers_bgr(image):
    """
    Convertit RGB en BGR (inversion des canaux).
    
    Paramètres:
    -----------
    image : numpy.ndarray
        Image RGB (H, W, 3) avec valeurs [0-255]
    
    Retourne:
    ---------
    numpy.ndarray : Image BGR
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être RGB (3 canaux)")
    
    return image[:, :, ::-1].copy()

def rgb_vers_image_indexee(image, nb_couleurs=256):
    """
    Convertit une image RGB (numpy array) en image indexée (palette) avec nb_couleurs couleurs max.
    Retourne une image numpy.ndarray de type uint8 (indices) et la palette (liste de couleurs RGB).
    """
    # Conversion numpy -> PIL
    img_pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    img_indexed = img_pil.convert("P", palette=Image.ADAPTIVE, colors=nb_couleurs)
    # Récupérer l'image indexée sous forme numpy (indices)
    indices = np.array(img_indexed)
    # Récupérer la palette (liste de couleurs RGB)
    palette = img_indexed.getpalette()[:nb_couleurs*3]
    palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)
    return indices, palette