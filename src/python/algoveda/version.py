"""
AlgoVeda version information
"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
__author__ = "AlgoVeda Team"
__email__ = "team@algoveda.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 AlgoVeda"

# Build information
__build__ = "20250910"
__commit__ = "main"

# API version
__api_version__ = "v1"

# Feature flags
FEATURES = {
    "gpu_acceleration": True,
    "cuda_support": True,
    "real_time_data": True,
    "options_trading": True,
    "portfolio_optimization": True,
    "backtesting": True,
    "paper_trading": True,
    "live_trading": True,
    "risk_management": True,
    "multi_broker": True,
    "web_interface": True,
    "api_access": True,
    "jupyter_integration": True,
    "cloud_deployment": True,
}

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get version tuple."""
    return __version_info__

def get_build_info():
    """Get build information."""
    return {
        "version": __version__,
        "build": __build__,
        "commit": __commit__,
        "api_version": __api_version__,
    }

def get_features():
    """Get enabled features."""
    return FEATURES.copy()

def is_feature_enabled(feature_name: str) -> bool:
    """Check if feature is enabled."""
    return FEATURES.get(feature_name, False)
