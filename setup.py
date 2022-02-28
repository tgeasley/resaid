#https://towardsdatascience.com/how-to-package-your-python-code-df5a7739ab2e
#https://packaging.python.org/tutorials/packaging-projects/

# Use the following commands to build
# python -m pip install -e .
# pip install -e .

import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resaid",
    version="0.0.3",
    author="Greg Easley",
    author_email="greg@easley.dev",
    description="Reservoir engineering tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    #python_requires=">=3.6",
)