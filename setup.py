"""
Setup script for the news headline topic classifier project.

This script helps set up the project environment and dependencies.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "News Headlines Topic Classifier using PyTorch"

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=1.9.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0"
        ]

setup(
    name="news-topic-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A PyTorch-based multiclass text classifier for news headline topics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/news_topic_classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
            "jupyterlab>=3.0.0",
        ],
        "nlp": [
            "nltk>=3.6.0",
            "spacy>=3.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "news-classifier-train=src.train:main",
            "news-classifier-predict=inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords="machine-learning pytorch nlp text-classification news deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/news_topic_classifier/issues",
        "Source": "https://github.com/yourusername/news_topic_classifier",
        "Documentation": "https://github.com/yourusername/news_topic_classifier#readme",
    },
)