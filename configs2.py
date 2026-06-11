"""
Configuration module - delegates to ConfigLoader for YAML-based config.
Maintains backward compatibility with existing code.
"""

import logging
from pathlib import Path
from config_loader import ConfigLoader, TrainingConfig as _TrainingConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Path = None) -> _TrainingConfig:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml in the script directory.
        
    Returns:
        TrainingConfig object
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    return ConfigLoader.load(config_path)


# Re-export for backward compatibility
TrainingConfig = _TrainingConfig
__all__ = ["TrainingConfig", "ConfigLoader", "load_config"]

