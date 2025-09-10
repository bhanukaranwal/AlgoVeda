"""
AlgoVeda - Complete Algorithmic Trading Platform

A comprehensive, production-ready algorithmic trading platform featuring:
- Ultra-low latency core engine (Rust + C++ + CUDA)
- Advanced options strategies with GPU pricing
- Real-time risk management and portfolio optimization
- High-performance WebSocket gateway
- Complete broker integrations
- 3D visualization and analytics
- Comprehensive backtesting framework
"""

from .__version__ import (
    __version__,
    __version_info__,
    __author__,
    __email__,
    __license__,
    __copyright__,
    get_version,
    get_version_info,
    get_build_info,
    get_features,
    is_feature_enabled,
)

# Core imports
from .core.base_strategy import BaseStrategy
from .core.strategy_manager import StrategyManager
from .core.backtest_engine import BacktestEngine
from .core.portfolio import Portfolio
from .core.position import Position

# Trading components
from .trading.order_manager import OrderManager, Order
from .trading.execution_engine import ExecutionEngine
from .trading.broker_interface import BrokerInterface

# Market data
from .market_data.data_provider import MarketDataProvider
from .market_data.realtime_feed import RealtimeDataFeed
from .market_data.historical_data import HistoricalDataManager

# Strategies
from .strategies.options.volatility_arbitrage import VolatilityArbitrageStrategy
from .strategies.options.market_making import OptionsMarketMakingStrategy
from .strategies.equity.momentum import MomentumStrategy
from .strategies.equity.mean_reversion import MeanReversionStrategy

# Risk management
from .risk.risk_manager import RiskManager
from .risk.portfolio_risk import PortfolioRiskCalculator
from .risk.var_calculator import VaRCalculator

# Analytics
from .analytics.performance import PerformanceAnalyzer
from .analytics.attribution import AttributionAnalyzer
from .analytics.optimization import PortfolioOptimizer

# Indicators
from .indicators.technical import TechnicalIndicators
from .indicators.custom import CustomIndicators

# Utilities
from .utils.config import Config
from .utils.logger import get_logger
from .utils.data_validation import validate_data
from .utils.performance import timer, profile

# Broker integrations
from .brokers.dhan import DhanBroker
from .brokers.zerodha import ZerodhaBroker
from .brokers.interactive_brokers import InteractiveBrokersBroker

# ML components
try:
    from .ml.features import FeatureEngineer
    from .ml.models import TradingModel
    from .ml.prediction import PricePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# GPU components
try:
    from .gpu.options_pricing import GPUOptionsPricer
    from .gpu.monte_carlo import GPUMonteCarlo
    from .gpu.risk_calculations import GPURiskCalculator
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Version check
import sys
if sys.version_info < (3, 9):
    raise RuntimeError("AlgoVeda requires Python 3.9 or higher")

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    "get_version",
    "get_version_info", 
    "get_build_info",
    "get_features",
    "is_feature_enabled",
    
    # Core classes
    "BaseStrategy",
    "StrategyManager", 
    "BacktestEngine",
    "Portfolio",
    "Position",
    
    # Trading
    "OrderManager",
    "Order",
    "ExecutionEngine",
    "BrokerInterface",
    
    # Market data
    "MarketDataProvider",
    "RealtimeDataFeed", 
    "HistoricalDataManager",
    
    # Strategies
    "VolatilityArbitrageStrategy",
    "OptionsMarketMakingStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    
    # Risk
    "RiskManager",
    "PortfolioRiskCalculator",
    "VaRCalculator",
    
    # Analytics  
    "PerformanceAnalyzer",
    "AttributionAnalyzer",
    "PortfolioOptimizer",
    
    # Indicators
    "TechnicalIndicators",
    "CustomIndicators",
    
    # Utilities
    "Config",
    "get_logger",
    "validate_data", 
    "timer",
    "profile",
    
    # Brokers
    "DhanBroker",
    "ZerodhaBroker", 
    "InteractiveBrokersBroker",
]

# Conditional exports
if ML_AVAILABLE:
    __all__.extend([
        "FeatureEngineer",
        "TradingModel", 
        "PricePredictor",
    ])

if GPU_AVAILABLE:
    __all__.extend([
        "GPUOptionsPricer",
        "GPUMonteCarlo",
        "GPURiskCalculator", 
    ])

# Initialize logging
logger = get_logger(__name__)
logger.info(f"AlgoVeda {__version__} initialized")
logger.info(f"Features available: ML={ML_AVAILABLE}, GPU={GPU_AVAILABLE}")

# Configuration
def configure(config_path: str = None, **kwargs):
    """
    Configure AlgoVeda with custom settings.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Configuration parameters
    """
    from .utils.config import configure_algoveda
    configure_algoveda(config_path, **kwargs)

def get_config():
    """Get current configuration."""
    from .utils.config import get_current_config
    return get_current_config()

# Add configuration functions to exports
__all__.extend(["configure", "get_config"])

# Platform info
def get_platform_info():
    """Get platform information."""
    import platform as plat
    import psutil
    import numpy as np
    
    info = {
        "algoveda_version": __version__,
        "python_version": sys.version,
        "platform": plat.platform(),
        "processor": plat.processor(),
        "architecture": plat.architecture(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy_version": np.__version__,
        "ml_available": ML_AVAILABLE,
        "gpu_available": GPU_AVAILABLE,
    }
    
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
            info["gpu_count"] = cp.cuda.runtime.getDeviceCount()
        except:
            pass
    
    return info

__all__.append("get_platform_info")

# Cleanup
del sys
