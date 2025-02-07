from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cnn_feature_extractor",
    version="0.1.0",
    author="ITU Perceptron",
    author_email="ituperceptron@gmail.com",
    description="Automatic CNN feature extraction and ML model comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ituperceptron/cnn_feature_extractor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=0.24.2",
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "tqdm>=4.50.0",
        "pillow>=8.0.0",
        "xgboost>=1.4.0",
        "lightgbm>=3.2.0",
    ],
) 