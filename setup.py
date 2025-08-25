#!/usr/bin/env python3
"""
Setup script for AI Hedge Fund project
"""

from setuptools import setup, find_packages

setup(
    name="ai-hedge-fund",
    version="1.0.0",
    description="AI-powered hedge fund with options data collection and analysis",
    author="AI Hedge Fund Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.13",
    install_requires=[
        # Core dependencies are managed by Poetry
        # This file is mainly for IDE recognition
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-cov",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
