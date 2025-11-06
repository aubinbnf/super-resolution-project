"""
Configuration centralisée pour le projet Super-Resolution
"""

from pathlib import Path
from typing import Literal

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "Real-ESRGAN"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Configuration Real-ESRGAN
class RealESRGANConfig:
    """Configuration pour Real-ESRGAN"""

    # Modèles disponibles
    MODELS = {
        "x4plus": {
            "path": MODELS_DIR / "RealESRGAN_x4plus.pth",
            "scale": 4,
            "num_blocks": 23,
            "description": "General purpose, best for natural images"
        },
        "anime": {
            "path": MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
            "num_blocks": 6,
            "description": "Optimized for anime/illustrations"
        }
    }

    # Paramètres d'inférence par défaut
    DEFAULT_SCALE = 4
    DEFAULT_DEVICE = "cuda"  # "cuda" ou "cpu"
    DEFAULT_TILE_SIZE = 0  # 0 = pas de tiling, >0 = découpe en tiles
    DEFAULT_TILE_PAD = 10
    DEFAULT_PRE_PAD = 0
    USE_FP16 = False  # True pour inference plus rapide mais moins précise

    # Architecture RRDBNet
    NUM_IN_CH = 3
    NUM_OUT_CH = 3
    NUM_FEAT = 64
    NUM_GROW_CH = 32


# Configuration Dataset
class DataConfig:
    """Configuration pour les datasets"""

    DIV2K_TRAIN_HR = DATA_DIR / "raw" / "DIV2K" / "DIV2K_train_HR"
    DIV2K_VALID_HR = DATA_DIR / "raw" / "DIV2K" / "DIV2K_valid_HR"

    DIV2K_TRAIN_LR = DATA_DIR / "processed" / "DIV2K" / "DIV2K_train_LR_bicubic"
    DIV2K_VALID_LR = DATA_DIR / "processed" / "DIV2K" / "DIV2K_valid_LR_bicubic"

    SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    MAX_IMAGE_SIZE = 4096  # Taille max en pixels (pour éviter OOM)


# Configuration API (à venir)
class APIConfig:
    """Configuration pour l'API FastAPI"""

    HOST = "0.0.0.0"
    PORT = 8000
    WORKERS = 4

    # Rate limiting
    RATE_LIMIT_FREE = 10  # Requêtes par heure
    RATE_LIMIT_PRO = 100

    # Upload limits
    MAX_UPLOAD_SIZE_MB = 10
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Processing
    PROCESSING_TIMEOUT = 60  # secondes
    QUEUE_MAX_SIZE = 100


# Configuration Métriques
class MetricsConfig:
    """Configuration pour le calcul des métriques"""

    COMPUTE_PSNR = True
    COMPUTE_SSIM = True
    COMPUTE_LPIPS = False  # Plus coûteux

    # PSNR config
    PSNR_DATA_RANGE = 255
    PSNR_CROP_BORDER = 4  # Crop borders pour éviter edge artifacts

    # SSIM config
    SSIM_MULTICHANNEL = True


# Configuration Outputs
class OutputsConfig:
    """Configuration pour les sorties"""

    SAVE_COMPARISON = True  # Sauvegarder avant/après côte à côte
    SAVE_METRICS = True     # Sauvegarder métriques en JSON
    OUTPUT_FORMAT = "png"   # Format de sortie
    OUTPUT_QUALITY = 95     # Qualité JPEG (si applicable)


# Environnement
ENV: Literal["development", "production"] = "development"

# Debug
DEBUG = ENV == "development"
VERBOSE = True
