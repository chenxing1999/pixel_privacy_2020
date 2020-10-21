import platform

from setuptools import find_packages, setup, find_namespace_packages



setup(
    name="pixel-privacy-biqa",
    version="v0.0.1",
    description="Pixel Privacy 2020",
    author="Hung V. Tran",
    url="https://github.com/xing1999/pixel_privacy_2020",
    packages=find_namespace_packages(
        exclude=["docs", "tests", "experiments", "scripts"]
    ),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=[],
    extras_require={},
)
