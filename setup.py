import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
DESCRIPTION = "GenKI"
PACKAGES = find_packages(exclude=("tests*",))
exec(open('GenKI/version.py').read())

INSTALL_REQUIRES = [
        "anndata==0.8.0",
        "matplotlib~=3.5.1",
        "numpy>=1.21.6",
        "pandas~=1.4.2",
        "ray>=1.11.0",
        "scanpy==1.9.1",
        "scipy~=1.8.0",
        "statsmodels~=0.13.2",
        "scikit_learn>=1.0.2",
        "torch==1.11.0",
        "tqdm~=4.64.0",
    ]

setup(
    name="GenKI",
    version=__version__,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yjgeno/GenKI",
    author="Yongjian Yang, TAMU",
    author_email="yjyang027@tamu.edu",
    license="MIT",
    keywords=[
        "neural network",
        "graph neural network",
        "variational graph neural network",
        "computational-biology",
        "single-cell",
        "gene knock-out",
        "gene regulatroy network",
    ],
    classifiers=[
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    ],
    python_requires='~=3.9.6',
    packages=PACKAGES,
    include_package_data=True, # MANIFEST
    install_requires=INSTALL_REQUIRES,
)