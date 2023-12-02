from setuptools import setup, find_packages

setup(
    name="streamlit_app",
    version="0.0.0a0",
    python_requires='>=3.11',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True
)