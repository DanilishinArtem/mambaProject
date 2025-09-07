from setuptools import setup, find_packages

setup(
    name="mamba-scan-adanilishin",
    version="0.1.0",
    description="Mamba chunk scan implementation",
    author="Danilishin Artem",
    packages=find_packages(),  # автоматически найдёт mamba_scan и triton
    install_requires=[
        "torch",
        "triton",  # если нужно
    ],
    python_requires=">=3.8",
)