from setuptools import setup, find_packages

setup(
    name="llmval",
    version="3.0.0",
    packages=find_packages(),
    package_data={
        "llmval": ["web/*", "web/**/*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "llmval = llmval.app:main",
        ],
    },
)
