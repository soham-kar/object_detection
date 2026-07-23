from setuptools import setup, find_packages

setup(
    name="wrdnet",
    version="0.1.0",
    description="Weather-Resilient Detection Unified Network",
    author="[Your Name]",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "ultralytics>=8.3.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "pycocotools>=2.0.7",
        "tensorboard>=2.14.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "einops>=0.7.0",
    ],
)
