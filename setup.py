from setuptools import setup, find_packages

setup(
    name="Stock Vision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # add other libraryyes installed
        "tensorflow",
        "numpy",
    ],
    author="Nabin Ghosh",
    description="A stock market analysis and prediction tool",
    python_requires=">=3.6",
)
