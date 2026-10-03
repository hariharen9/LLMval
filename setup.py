from setuptools import setup, find_packages

setup(
    name="llmval",
    version="1.0.2",
    author="Hariharen",
    url="https://hariharen.site",
    packages=find_packages(),
    package_data={
        "llmval": ["web/*", "web/**/*", "data/*", "data/**/*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "llmval = llmval.app:main",
        ],
    },
)
