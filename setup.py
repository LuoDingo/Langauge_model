import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


__version__ = "1.3.1"


setuptools.setup(
    name='sentence-suggestion',
    version=__version__,
    author='Kei Nemoto, Steven Alshheimer',
    author_email='',
    description='A package that contains sentence suggestion models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LuoDingo/Langauge_model',
    packages=setuptools.find_packages(exclude=('data', 'test')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'fuzzywuzzy',
        'pandas',
        'numpy',
        'torch>=1.4',
        'torchtext>=0.5',
        'dill>=0.3',
        'tqdm>=4.31'
    ],
    python_requires='>=3.7',
)
