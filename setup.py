from setuptools import setup, find_packages

setup(
    name="atdl-swss",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib>=3.10.7",
        "tensorflow>=2.20.0",
        "tqdm>=4.66.0",
        "pillow>=10.0.0",
        "imageio>=2.9.0",
    ],
    python_requires=">=3.13",
)