"""Configuration loading utilities for WRDNet.

Supports nested YAML configs with flat access via getattr.
Example: config.training.lr or getattr(config, 'lr') both work
because the config is flattened after loading.
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """
    Config container with both nested and flat dot-notation access.

    Nested YAML keys like `training.lr` are accessible as:
      - config.training.lr  (nested)
      - getattr(config, 'lr')  (flat — searches all levels)
    """

    def __init__(self, config_dict: Dict[str, Any], _prefix: str = ''):
        object.__setattr__(self, '_config', config_dict)
        object.__setattr__(self, '_flat', {})

        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively create nested Config
                nested = Config(value, _prefix=f'{_prefix}{key}.')
                setattr(self, key, nested)
                # Merge nested flat keys into this level's flat dict
                self._flat.update(nested._flat)
                # Also store this key itself (so config.model works)
                self._flat[key] = nested
            else:
                setattr(self, key, value)
                self._flat[key] = value

    def __getattr__(self, name: str) -> Any:
        """Fallback: search flat dict for keys from any nesting level."""
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._flat:
            return self._flat[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Search flat keys first, then nested dict."""
        if key in self._flat:
            return self._flat[key]
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """Load a YAML config file and return a Config object."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """Merge override dict into base config (shallow merge)."""
    merged = base_config.to_dict().copy()
    merged.update(override_dict)
    return Config(merged)
