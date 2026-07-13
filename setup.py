from setuptools import setup, find_packages

setup(
    name="medical_CTTA",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "transformers>=4.30.0",
        "huggingface_hub>=0.14.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pillow>=9.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.6.0",
        "jupyter>=1.0.0",
    ],
)