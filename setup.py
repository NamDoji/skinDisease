from setuptools import setup, find_packages

setup(
    name="dermnet",
    version="0.1.0",
    description="Skin disease image classification using EfficientNet-B0 fine-tuned on DermNet",
    author="duongphamminhdung",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "timm>=0.9",
        "albumentations>=1.3",
        "opencv-python>=4.7",
        "Pillow>=9.0",
        "scikit-learn>=1.2",
        "pandas>=1.5",
        "numpy>=1.23",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "omegaconf>=2.3",
        "grad-cam>=1.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1",
        ],
        "notebook": [
            "jupyter",
            "notebook",
            "ipykernel",
            "torchinfo",
        ],
    },
)
