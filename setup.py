from setuptools import setup, find_packages

setup(
    name="cvpr_main",
    version="0.1",
    packages=find_packages(),     # <-- this will pick up the inner cvpr_main/ folder
)