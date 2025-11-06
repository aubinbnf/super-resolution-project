"""
Utilitaires pour le traitement d'images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def load_image(path: str, mode: str = "RGB") -> np.ndarray:
    """
    Charge une image depuis un chemin

    Args:
        path: Chemin de l'image
        mode: "RGB" ou "BGR"

    Returns:
        Image en numpy array
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Impossible de charger l'image : {path}")

    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_image(img: np.ndarray, path: str, mode: str = "RGB") -> None:
    """
    Sauvegarde une image

    Args:
        img: Image en numpy array
        path: Chemin de destination
        mode: "RGB" ou "BGR"
    """
    # Créer le dossier parent si nécessaire
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), img)


def resize_image(img: np.ndarray, scale: float, interpolation: str = "bicubic") -> np.ndarray:
    """
    Redimensionne une image

    Args:
        img: Image source
        scale: Facteur de redimensionnement
        interpolation: "nearest", "bilinear", "bicubic", "lanczos"

    Returns:
        Image redimensionnée
    """
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }

    interp = interpolation_map.get(interpolation, cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    new_size = (int(w * scale), int(h * scale))

    return cv2.resize(img, new_size, interpolation=interp)


def create_comparison(
    img_original: np.ndarray,
    img_upscaled: np.ndarray,
    img_baseline: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Crée une comparaison visuelle avant/après

    Args:
        img_original: Image originale (ground truth)
        img_upscaled: Image après upscaling
        img_baseline: Image baseline (ex: bicubic) optionnelle
        save_path: Chemin pour sauvegarder la comparaison
    """
    n_images = 3 if img_baseline is not None else 2
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))

    if n_images == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]

    # Original
    axes[0].imshow(img_original)
    axes[0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Baseline (si fournie)
    if img_baseline is not None:
        axes[1].imshow(img_baseline)
        axes[1].set_title("Baseline (Bicubic)", fontsize=14, fontweight='bold', color='orange')
        axes[1].axis('off')
        upscaled_idx = 2
    else:
        upscaled_idx = 1

    # Upscaled
    axes[upscaled_idx].imshow(img_upscaled)
    axes[upscaled_idx].set_title("Real-ESRGAN", fontsize=14, fontweight='bold', color='green')
    axes[upscaled_idx].axis('off')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison saved: {save_path}")

    plt.show()


def crop_center(img: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Crop le centre d'une image

    Args:
        img: Image source
        crop_size: Taille du crop (carré)

    Returns:
        Image croppée
    """
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2

    y1 = max(0, center_y - crop_size // 2)
    y2 = min(h, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(w, center_x + crop_size // 2)

    return img[y1:y2, x1:x2]


def validate_image_size(img: np.ndarray, max_size: int = 4096) -> bool:
    """
    Vérifie que l'image ne dépasse pas une taille maximale

    Args:
        img: Image à vérifier
        max_size: Taille max en pixels (hauteur ou largeur)

    Returns:
        True si valide, False sinon
    """
    h, w = img.shape[:2]
    return h <= max_size and w <= max_size


def get_image_info(img: np.ndarray) -> dict:
    """
    Récupère les informations d'une image

    Args:
        img: Image

    Returns:
        Dictionnaire avec les infos
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1

    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(img.dtype),
        "size_mb": img.nbytes / (1024 ** 2)
    }
