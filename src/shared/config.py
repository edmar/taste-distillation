"""
Configuration loader for DSPy taste prediction models.

Provides centralized configuration management using YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration manager for DSPy models and scripts."""
    
    def __init__(self, config_name: str = "evaluate", config_path: Optional[str] = None, environment: str = "default"):
        """
        Initialize configuration.
        
        Args:
            config_name: Name of config file (evaluate, train, etc.)
            config_path: Path to config file. If None, uses default location.
            environment: Environment name (default, development, production, testing)
        """
        if config_path is None:
            # Find config directory - check multiple possible locations
            current_dir = Path(__file__).parent
            config_locations = [
                # Check in src/dspy_favorite/config/ (new location)
                current_dir.parent / "dspy_favorite" / "config",
                # Check in project root config/ (old location)
                current_dir.parent.parent / "config",
            ]
            
            config_path = None
            for location in config_locations:
                if location.exists():
                    config_path = location / f"{config_name}.yaml"
                    if config_path.exists():
                        break
            
            if config_path is None or not config_path.exists():
                raise FileNotFoundError(f"Could not find {config_name}.yaml in any of these locations: {config_locations}")
        
        self.config_path = Path(config_path)
        self.config_name = config_name
        self.environment = environment
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Start with default configuration
        result = config.get('default', {}).copy()
        
        # Apply environment-specific overrides
        if self.environment != "default" and 'environments' in config:
            env_config = config['environments'].get(self.environment, {})
            result = self._deep_merge(result, env_config)
        
        return result
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_llm_name(self, llm_override: Optional[str] = None) -> str:
        """Get LLM model name with optional override."""
        if llm_override:
            # Check if this is a named LLM config
            llm_configs = self._get_llm_configs()
            if llm_override in llm_configs:
                return llm_configs[llm_override].get('model_name', llm_override)
            return llm_override
        return self._config.get('llm', {}).get('model_name', 'openai/gpt-4o-mini')
    
    def get_llm_config(self, llm_override: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration with optional override."""
        if llm_override:
            # Check if this is a named LLM config
            llm_configs = self._get_llm_configs()
            if llm_override in llm_configs:
                return llm_configs[llm_override]
            
            # Check if this is a full OpenAI model name (e.g., "openai/o3")
            # and try to match with a named config
            if llm_override.startswith('openai/'):
                model_suffix = llm_override[7:]  # Remove "openai/" prefix
                if model_suffix in llm_configs:
                    return llm_configs[model_suffix]
        return self._config.get('llm', {})
    
    def _get_llm_configs(self) -> Dict[str, Any]:
        """Get the llms section from the full config."""
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        return full_config.get('llms', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get('evaluation', {})
    
    def get_datasets_config(self) -> Dict[str, Any]:
        """Get datasets configuration."""
        return self._config.get('datasets', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get('paths', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_llm_config_by_name(self, llm_name: str) -> Dict[str, Any]:
        """Get specific LLM configuration by name."""
        # First try to load full config again to get llms section
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        llms = full_config.get('llms', {})
        if llm_name in llms:
            return llms[llm_name]
        return {}


def load_config(config_name: str = "evaluate", environment: str = None) -> Config:
    """
    Load configuration for current environment.
    
    Args:
        config_name: Name of config file (evaluate, train, etc.)
        environment: Environment name. If None, uses TASTE_ENV or 'default'
    
    Returns:
        Config object
    """
    if environment is None:
        environment = os.getenv('TASTE_ENV', 'default')
    
    return Config(config_name=config_name, environment=environment)


# Global config instances for different configs
_global_configs = {}

def get_config(config_name: str = "evaluate") -> Config:
    """Get global config instance (singleton pattern)."""
    global _global_configs
    if config_name not in _global_configs:
        _global_configs[config_name] = load_config(config_name)
    return _global_configs[config_name]