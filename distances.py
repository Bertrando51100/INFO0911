import numpy as np


def distance_l1(vecteur1, vecteur2):
    """
    Calcule la distance L1 (Manhattan) entre deux vecteurs.
    
    Formule : d = Σ|v1[i] - v2[i]|
    
    Paramètres:
    -----------
    vecteur1 : numpy.ndarray
        Premier vecteur descripteur
    vecteur2 : numpy.ndarray
        Deuxième vecteur descripteur
    
    Retourne:
    ---------
    float : Distance L1 entre les deux vecteurs
    
    Exemple:
    --------
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> distance_l1(v1, v2)
    9.0
    """
    if vecteur1.shape != vecteur2.shape:
        raise ValueError(f"Les vecteurs doivent avoir la même taille: {vecteur1.shape} vs {vecteur2.shape}")
    
    return np.sum(np.abs(vecteur1 - vecteur2))


def distance_l2(vecteur1, vecteur2):
    """
    Calcule la distance L2 (Euclidienne) entre deux vecteurs.
    
    Formule : d = √(Σ(v1[i] - v2[i])²)
    
    Paramètres:
    -----------
    vecteur1 : numpy.ndarray
        Premier vecteur descripteur
    vecteur2 : numpy.ndarray
        Deuxième vecteur descripteur
    
    Retourne:
    ---------
    float : Distance L2 entre les deux vecteurs
    
    Exemple:
    --------
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> distance_l2(v1, v2)
    5.196152422706632
    """
    if vecteur1.shape != vecteur2.shape:
        raise ValueError(f"Les vecteurs doivent avoir la même taille: {vecteur1.shape} vs {vecteur2.shape}")
    
    return np.sqrt(np.sum((vecteur1 - vecteur2) ** 2))


def distance_cosinus(vecteur1, vecteur2):
    """
    Calcule la distance cosinus entre deux vecteurs.
    
    Formule : d = 1 - (v1·v2) / (||v1|| * ||v2||)
    
    Paramètres:
    -----------
    vecteur1 : numpy.ndarray
        Premier vecteur descripteur
    vecteur2 : numpy.ndarray
        Deuxième vecteur descripteur
    
    Retourne:
    ---------
    float : Distance cosinus entre les deux vecteurs (0 = identiques, 2 = opposés)
    """
    if vecteur1.shape != vecteur2.shape:
        raise ValueError(f"Les vecteurs doivent avoir la même taille: {vecteur1.shape} vs {vecteur2.shape}")
    
    # Produit scalaire
    dot_product = np.dot(vecteur1, vecteur2)
    
    # Normes
    norm1 = np.linalg.norm(vecteur1)
    norm2 = np.linalg.norm(vecteur2)
    
    # Éviter la division par zéro
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    # Similarité cosinus
    similarite = dot_product / (norm1 * norm2)
    
    # Distance cosinus
    return 1 - similarite


def distance_chi2(vecteur1, vecteur2):
    """
    Calcule la distance Chi-carré entre deux vecteurs (utile pour les histogrammes).
    
    Formule : d = 0.5 * Σ((v1[i] - v2[i])² / (v1[i] + v2[i]))
    
    Paramètres:
    -----------
    vecteur1 : numpy.ndarray
        Premier vecteur descripteur
    vecteur2 : numpy.ndarray
        Deuxième vecteur descripteur
    
    Retourne:
    ---------
    float : Distance Chi-carré
    """
    if vecteur1.shape != vecteur2.shape:
        raise ValueError(f"Les vecteurs doivent avoir la même taille: {vecteur1.shape} vs {vecteur2.shape}")
    
    # Somme des vecteurs
    somme = vecteur1 + vecteur2
    
    # Éviter la division par zéro
    mask = somme > 0
    
    # Calcul de la distance
    distance = 0.0
    if np.any(mask):
        numerateur = (vecteur1[mask] - vecteur2[mask]) ** 2
        denominateur = somme[mask]
        distance = 0.5 * np.sum(numerateur / denominateur)
    
    return distance


def distance_intersection(vecteur1, vecteur2):
    """
    Calcule la distance par intersection d'histogrammes.
    
    Formule : d = 1 - Σmin(v1[i], v2[i])
    
    Paramètres:
    -----------
    vecteur1 : numpy.ndarray
        Premier vecteur descripteur (normalisé)
    vecteur2 : numpy.ndarray
        Deuxième vecteur descripteur (normalisé)
    
    Retourne:
    ---------
    float : Distance (0 = identiques, 1 = complètement différents)
    """
    if vecteur1.shape != vecteur2.shape:
        raise ValueError(f"Les vecteurs doivent avoir la même taille: {vecteur1.shape} vs {vecteur2.shape}")
    
    intersection = np.sum(np.minimum(vecteur1, vecteur2))
    return 1 - intersection