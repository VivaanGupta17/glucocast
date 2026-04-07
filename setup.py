"""
GlucoCast: Deep Learning for Continuous Glucose Monitoring Prediction.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="glucocast",
    version="1.0.0",
    author="GlucoCast Contributors",
    description="Multi-horizon glucose prediction using Temporal Fusion Transformers and LSTM for CGM data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/glucocast",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "lxml>=4.9.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.22.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "glucocast-train=scripts.train:main",
            "glucocast-eval=scripts.evaluate:main",
            "glucocast-predict=scripts.predict:main",
        ],
    },
    keywords=[
        "glucose", "cgm", "diabetes", "machine-learning",
        "time-series", "transformer", "insulin", "healthcare"
    ],
)
