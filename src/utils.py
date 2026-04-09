"""
Utilitaires : chargement de config, logging, helpers.
"""

import yaml
import logging
from pathlib import Path

# ── Chemins du projet ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
OUTPUTS = PROJECT_ROOT / "outputs"

# ── Config ─────────────────────────────────────────────────────────
def load_config(path=None):
    """Charge le fichier config.yaml."""
    path = path or CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── Logging ────────────────────────────────────────────────────────
def setup_logger(name="football", level=logging.INFO):
    """Configure un logger avec format lisible."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
