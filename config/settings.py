import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PathsConfig(BaseModel):
    """Configuration for application directories."""
    data_dir: Path = Field(default=Path("./data"), description="Directory for data storage")
    model_dir: Path = Field(default=Path("./saved_model"), description="Directory for saved models")
    log_dir: Path = Field(default=Path("./logs"), description="Directory for application logs")


class ModelConfig(BaseModel):
    """Configuration for the sentiment analysis model."""
    model_name: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest", description="HuggingFace model name")
    max_length: int = Field(default=128, description="Maximum sequence length for tokenization")
    batch_size: int = Field(default=16, description="Batch size for model inference")


class Settings(BaseSettings):
    """Main application settings."""
    paths: PathsConfig
    model: ModelConfig


def _load_yaml_config(filepath: Path) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    
    Args:
        filepath (Path): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache()
def get_settings() -> Settings:
    """
    Instantiates and caches the application settings.
    Reads from the config.yaml file and validates types.
    Automatically creates necessary directories if they do not exist.
    
    Returns:
        Settings: Evaluated application settings.
    """
    config_path = Path("config/config.yaml")
    
    # Load settings from the YAML file
    yaml_data = _load_yaml_config(config_path)
    
    settings = Settings(**yaml_data)
    
    # Create required directories automatically
    for dir_path in [settings.paths.data_dir, settings.paths.model_dir, settings.paths.log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return settings
