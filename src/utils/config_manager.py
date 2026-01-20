"""
Configuration Management Module

This module provides centralized configuration management for the stress detection system.
It handles loading, validation, and access to configuration parameters from YAML files.

Author: Research Team
Date: 2024
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'data': {
        'raw_data_dir': 'data/raw',
        'processed_data_dir': 'data/processed',
        'wesad_dir': 'data/wesad',
        'empatica_dir': 'data/empatica_e4_stress'
    },
    'preprocessing': {
        'ppg': {
            'filter_type': 'chebyshev2',
            'filter_order': 4,
            'lowcut': 0.5,
            'highcut': 5.0,
            'artifact_threshold': 20,
            'interpolation': False,
            'min_rr_interval': 500,
            'max_rr_interval': 1200
        },
        'windowing': {
            'window_size': 300,
            'overlap': 150,
            'min_valid_data': 0.85
        }
    },
    'models': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'class_weight': 'balanced',
            'random_state': 42
        }
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'console': {
            'enabled': True,
            'level': 'INFO'
        },
        'file': {
            'enabled': True,
            'path': 'logs/pipeline.log',
            'max_bytes': 1048576,
            'backup_count': 5,
            'level': 'INFO'
        }
    }
}


class ConfigManager:
    """
    Centralized configuration management system.
    
    This class provides methods to load, validate, and access configuration
    parameters for the stress detection system.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Path to configuration file. If None, searches for default locations.
        """
        self.config_path = self._find_config_file(config_path)
        self.config = self._load_config()
        self._validate_config()
        
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Path:
        """
        Find the configuration file in default locations.
        
        Parameters
        ----------
        config_path : str or Path, optional
            Explicit path to configuration file
            
        Returns
        -------
        Path
            Path to configuration file
        """
        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                return path
            else:
                logger.warning(f"Specified config file {path} not found")
        
        # Search in default locations
        possible_paths = [
            Path('config/config.yaml'),
            Path('../config/config.yaml'),
            Path('../../config/config.yaml'),
            Path('./config.yaml')
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        logger.warning("No configuration file found, using defaults")
        return None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        if self.config_path is None or not self.config_path.exists():
            logger.info("Using default configuration")
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Merge with defaults to ensure all required keys exist
            merged_config = self._merge_configs(DEFAULT_CONFIG, config)
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_path}: {str(e)}")
            logger.info("Falling back to default configuration")
            return DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge user configuration with defaults.
        
        Parameters
        ----------
        default : Dict[str, Any]
            Default configuration
        user : Dict[str, Any]
            User configuration
            
        Returns
        -------
        Dict[str, Any]
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If configuration contains invalid values
        """
        # Validate data paths
        try:
            data_config = self.config.get('data', {})
            for key, path in data_config.items():
                if path and not Path(path).parent.exists():
                    logger.warning(f"Parent directory for {key} ({path}) does not exist")
        except Exception as e:
            logger.warning(f"Path validation warning: {str(e)}")
        
        # Validate preprocessing parameters
        ppg_config = self.config.get('preprocessing', {}).get('ppg', {})
        
        # Validate filter parameters
        lowcut = ppg_config.get('lowcut', 0.5)
        highcut = ppg_config.get('highcut', 5.0)
        
        if lowcut >= highcut:
            raise ValueError(f"Invalid filter frequencies: lowcut ({lowcut}) >= highcut ({highcut})")
        
        if lowcut < 0 or highcut < 0:
            raise ValueError("Filter frequencies must be positive")
        
        # Validate RR interval limits
        min_rr = ppg_config.get('min_rr_interval', 500)
        max_rr = ppg_config.get('max_rr_interval', 1200)
        
        if min_rr >= max_rr:
            raise ValueError(f"Invalid RR limits: min_rr ({min_rr}) >= max_rr ({max_rr})")
        
        # Validate windowing parameters
        window_config = self.config.get('preprocessing', {}).get('windowing', {})
        window_size = window_config.get('window_size', 300)
        overlap = window_config.get('overlap', 150)
        
        if overlap >= window_size:
            raise ValueError(f"Window overlap ({overlap}) >= window size ({window_size})")
        
        logger.info("Configuration validation completed successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key in dot notation (e.g., 'preprocessing.ppg.lowcut')
        default : Any, optional
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key in dot notation
        value : Any
            Value to set
        """
        keys = key.split('.')
        config_dict = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        # Set the value
        config_dict[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_data_paths(self) -> Dict[str, Path]:
        """
        Get all data paths as Path objects.
        
        Returns
        -------
        Dict[str, Path]
            Dictionary of data paths
        """
        data_config = self.config.get('data', {})
        return {key: Path(path) for key, path in data_config.items()}
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.
        
        Returns
        -------
        Dict[str, Any]
            Preprocessing configuration
        """
        return self.config.get('preprocessing', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
            
        Returns
        -------
        Dict[str, Any]
            Model configuration
        """
        models_config = self.config.get('models', {})
        return models_config.get(model_name, {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Return logging configuration dictionary."""
        return self.config.get('logging', {})
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Output path for configuration file. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        if output_path is None:
            raise ValueError("No output path specified and no original config file")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def create_dirs(self) -> None:
        """
        Create all directories specified in configuration.
        """
        data_paths = self.get_data_paths()
        
        for name, path in data_paths.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")
            except Exception as e:
                logger.warning(f"Failed to create directory {path}: {str(e)}")
        
        # Create results directories
        results_dir = Path('results')
        for subdir in ['models', 'figures', 'reports']:
            (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def __getitem__(self, key: str) -> Any:
        """
        Enable dictionary-style access to configuration.
        
        Parameters
        ----------
        key : str
            Configuration key
            
        Returns
        -------
        Any
            Configuration value
        """
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Enable dictionary-style setting of configuration.
        
        Parameters
        ----------
        key : str
            Configuration key
        value : Any
            Value to set
        """
        self.set(key, value)


# Global configuration instance
_config_manager = None


def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns
    -------
    ConfigManager
        Global configuration manager
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Load configuration and set as global instance.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file
        
    Returns
    -------
    ConfigManager
        Configuration manager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


if __name__ == "__main__":
    # Example usage and testing
    try:
        # Test configuration loading
        config = ConfigManager()
        
        print("Configuration Test Results:")
        print(f"Config loaded from: {config.config_path}")
        
        # Test getting values
        print(f"PPG lowcut frequency: {config.get('preprocessing.ppg.lowcut')}")
        print(f"Random Forest n_estimators: {config.get('models.random_forest.n_estimators')}")
        
        # Test data paths
        data_paths = config.get_data_paths()
        print(f"Data paths: {list(data_paths.keys())}")
        
        # Test model config
        rf_config = config.get_model_config('random_forest')
        print(f"Random Forest config: {rf_config}")
        
        # Test directory creation
        config.create_dirs()
        print("Directories created successfully")
        
    except Exception as e:
        print(f"Configuration test failed: {str(e)}")