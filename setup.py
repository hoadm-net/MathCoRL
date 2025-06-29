"""
Setup script for MINT - Mathematical Intelligence Library
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "MINT - Mathematical Intelligence Library with FPP, CoT, PoT, and Zero-Shot prompting methods"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="mint",
    version="0.3.0",
    author="MathCoRL Team",
    author_email="",
    description="Mathematical Intelligence Library with FPP, CoT, PoT, and Zero-Shot prompting methods",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mathcorl/mint",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        'mint': ['../templates/*.txt'],
    },
    entry_points={
        'console_scripts': [
            'mint-fpp=mint.cli:main',
        ],
    },
    keywords="mathematics, ai, llm, function-prototype-prompting, chain-of-thought, program-of-thoughts, zero-shot, mathematical-reasoning",
    project_urls={
        "Bug Reports": "https://github.com/mathcorl/mint/issues",
        "Source": "https://github.com/mathcorl/mint",
    },
) 