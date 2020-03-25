import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='sentence-suggestion',
    version='0.1',
    author='Kei Nemoto, Steven Alshheimer (equal contribution)',
    author_email='',
    description='A package that contains sentence suggestion models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LuoDingo/Langauge_model',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
