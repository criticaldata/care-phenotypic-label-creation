from setuptools import setup, find_packages

setup(
    name="care-phenotype-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",  # For statistical analysis
        "matplotlib>=3.5.0",    # For visualization
        "seaborn>=0.11.0",      # For statistical visualizations
    ],
    author="MIT Critical Data",
    author_email="criticaldata@mit.edu",
    description="A package for creating objective care phenotype labels based on observable care patterns",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MIT-LCP/care-phenotypic-label-creation",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
) 