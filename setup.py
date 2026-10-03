from setuptools import setup, find_packages

setup(
    name="best-model-today",
    version="3.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "best-model-today = best_model_today.app:main",
            "best_model_today = best_model_today.app:main",
        ],
    },
)
