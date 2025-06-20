from setuptools import setup, find_packages

setup(
    name="vqa_sdes",
    version="0.1.0",
    description="Variational Quantum Algorithm for S-DES Key Recovery",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pylatexenc",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "vqa-sdes=main:main"
        ]
    },
)
