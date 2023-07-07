""""Set up for Systema Smart App
"""

import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name=f"ozsoftcon_mlflow",
    version="0.1.0",
    author="OZSoftCon",
    author_email="ozsoftcon@gmail.com",
    description="MLFlow Sample code",
    long_description="",
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    #project_urls={
    #    "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    #},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=[
        'ozsoftcon',
        'ozsoftcon.mlflow_wrap',
        'ozsoftcon.utils'
        'ozsoftcon.ml'
    ]),
    python_requires=">=3.9, <4.0"
)