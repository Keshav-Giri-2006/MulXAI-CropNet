"""Setup file for MulXAI-CropNet."""

from setuptools import setup, find_packages

setup(
    name='mulxai-cropnet',
    version='0.1.0',
    description='Tomato disease classification using deep learning',
    author='MulXAI Research Team',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'albumentations>=1.3.0',
        'opencv-python>=4.7.0',
        'Pillow>=9.5.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tqdm>=4.65.0',
    ],
)
