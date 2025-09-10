"""
Setup script for AlgoVeda Python package
Builds Cython extensions and creates installable package
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import os
import platform
from pathlib import Path

# Read version
def get_version():
    version_file = Path(__file__).parent / "algoveda" / "__version__.py"
    version_dict = {}
    with open(version_file) as f:
        exec(f.read(), version_dict)
    return version_dict["__version__"]

# Read long description
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Compiler flags
extra_compile_args = ["-O3", "-Wall", "-Wextra"]
extra_link_args = []

if platform.system() == "Darwin":  # macOS
    extra_compile_args.extend(["-std=c++17", "-stdlib=libc++"])
    extra_link_args.extend(["-stdlib=libc++"])
elif platform.system() == "Linux":
    extra_compile_args.extend(["-std=c++17", "-march=native", "-ffast-math"])
elif platform.system() == "Windows":
    extra_compile_args = ["/O2", "/std:c++17"]

# Include directories
include_dirs = [
    numpy.get_include(),
    "../../cpp/include",
    "../../cuda/include",
    str(Path(__file__).parent / "algoveda" / "cython" / "include"),
]

# CUDA library paths
cuda_lib_dirs = []
cuda_libraries = []

if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    cuda_lib_dirs.extend([
        os.path.join(cuda_path, "lib64"),
        os.path.join(cuda_path, "lib"),
    ])
    cuda_libraries.extend(["cuda", "cudart", "cublas", "curand"])
    include_dirs.append(os.path.join(cuda_path, "include"))

# Cython extensions
extensions = [
    # Core backtesting engine
    Extension(
        "algoveda.core.backtest_engine_cy",
        ["algoveda/core/backtest_engine_cy.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    # Options pricing calculations
    Extension(
        "algoveda.calculations.options_pricing.black_scholes_cy",
        ["algoveda/calculations/options_pricing/black_scholes_cy.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    # Technical indicators
    Extension(
        "algoveda.indicators.technical_indicators_cy",
        ["algoveda/indicators/technical_indicators_cy.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    # Portfolio calculations
    Extension(
        "algoveda.portfolio.portfolio_calculations_cy",
        ["algoveda/portfolio/portfolio_calculations_cy.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    # Risk calculations
    Extension(
        "algoveda.risk.risk_calculations_cy",
        ["algoveda/risk/risk_calculations_cy.pyx"],
        include_dirs=include_dirs + cuda_lib_dirs,
        libraries=cuda_libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    
    # Market data processing
    Extension(
        "algoveda.market_data.data_processing_cy",
        ["algoveda/market_data/data_processing_cy.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

# Cythonize extensions
extensions = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "embedsignature": True,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
    },
    annotate=True,  # Generate HTML annotation files
)

# Requirements
install_requires = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "dash>=2.10.0",
    "streamlit>=1.25.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "redis>=4.5.0",
    "celery>=5.3.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
    "websockets>=11.0.0",
    "python-socketio>=5.8.0",
    "pyjwt>=2.8.0",
    "cryptography>=41.0.0",
    "click>=8.1.0",
    "pyyaml>=6.0",
    "toml>=0.10.0",
    "python-dotenv>=1.0.0",
    "loguru>=0.7.0",
    "rich>=13.4.0",
    "typer>=0.9.0",
    "httpx>=0.24.0",
    "polars>=0.18.0",
    "pyarrow>=12.0.0",
    "numba>=0.57.0",
    "talib>=0.4.26",
    "yfinance>=0.2.18",
    "alpha_vantage>=2.3.1",
    "polygon-api-client>=1.12.0",
    "jupyter>=1.0.0",
    "jupyterlab>=4.0.0",
    "ipywidgets>=8.0.0",
    "notebook>=7.0.0",
]

extras_require = {
    "gpu": [
        "cupy-cuda11x>=12.0.0",
        "numba[cuda]>=0.57.0",
    ],
    "ml": [
        "tensorflow>=2.13.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "lightgbm>=4.0.0",
        "xgboost>=1.7.0",
        "catboost>=1.2.0",
        "optuna>=3.2.0",
        "hyperopt>=0.2.7",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "pre-commit>=3.3.0",
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
        "sphinx-autodoc-typehints>=1.24.0",
        "sphinx-copybutton>=0.5.2",
    ],
    "testing": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "pytest-asyncio>=0.21.0",
        "pytest-xdist>=3.3.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.82.0",
        "factory-boy>=3.3.0",
        "faker>=19.3.0",
    ],
}

# All extras
extras_require["all"] = list(set().union(*extras_require.values()))

setup(
    name="algoveda",
    version=get_version(),
    description="Complete Algorithmic Trading Platform",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="AlgoVeda Team",
    author_email="team@algoveda.com",
    url="https://github.com/algoveda/platform",
    project_urls={
        "Documentation": "https://docs.algoveda.com",
        "Source": "https://github.com/algoveda/platform",
        "Tracker": "https://github.com/algoveda/platform/issues",
    },
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "algoveda": [
            "data/*.csv",
            "data/*.json",
            "config/*.yaml",
            "config/*.json",
            "templates/*.py",
            "templates/*.yaml",
            "static/*",
        ],
    },
    ext_modules=extensions,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
        "Programming Language :: C++",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "algorithmic trading",
        "quantitative finance",
        "backtesting",
        "options trading",
        "high frequency trading",
        "financial markets",
        "trading strategies",
        "portfolio optimization",
        "risk management",
        "market data",
    ],
    entry_points={
        "console_scripts": [
            "algoveda=algoveda.cli.main:main",
            "algoveda-backtest=algoveda.cli.backtest:main",
            "algoveda-deploy=algoveda.cli.deploy:main",
            "algoveda-data=algoveda.cli.data:main",
        ],
    },
    cmdclass={},
)
