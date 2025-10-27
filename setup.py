from setuptools import setup, find_packages

setup(
    name="ATDL-SoftWeightSharing",
    version="0.1.0",
    description="A tutorial on Soft weight-sharing for Neural Network compression",
    author="Karen Ullrich, adapted for Python 3.13+ and TensorFlow 2+",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.13",
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "imageio>=2.9.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "atdl-tutorial=ATDLSoftWeightSharing.main:main",
        ],
    },
)