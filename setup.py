import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="AutoDiff-StanAndyJohn", # Replace with your own username
    version="0.1.1",
    author="Harvard CS207 Group 33",
    author_email="johnathan_jiang@harvard.edu",
    description="autodifferentiation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StanAndyJohn/cs207-FinalProject",
    install_requires = [
        'numpy>=1.15.2',
        'pythonds>=1.2.1',
        'pytest>=3.4.2',
        'pytest-cov>=2.5.1',
        'pytest-dependency>=0.4',
        'matplotlib==3.1.1'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
