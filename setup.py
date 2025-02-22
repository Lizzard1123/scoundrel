from setuptools import setup, find_packages

setup(
    name="scoundrel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rich>=13.0.0",
    ],
    entry_points={
        'console_scripts': [
            'scoundrel=scoundrel.main:main',
        ],
    },
)