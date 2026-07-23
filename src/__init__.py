"""WRDNet: Weather-Resilient Detection Unified Network.

Joint defogging + detection + depth estimation for adverse weather.
"""

from .models.wrnet import WRDNet
from .utils.config import Config, load_config

__all__ = ['WRDNet', 'Config', 'load_config']

