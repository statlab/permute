"""
Permutation tests and confidence sets for Python
"""

import os


DISTNAME = "permute"
DESCRIPTION = "Permutation tests and confidence sets for Python"
AUTHOR = "K. Jarrod Millman, Kellie Ottoboni, and Philip B. Stark"
AUTHOR_EMAIL = "permute@googlegroups.com"
URL = "http://statlab.github.io/permute/"
LICENSE = "BSD License"
DOWNLOAD_URL = "http://github.com/statlab/permute"
VERSION = "0.2.alpha1"


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


INSTALL_REQUIRES = parse_requirements_file("requirements/default.txt")
TESTS_REQUIRE = parse_requirements_file("requirements/test.txt")

with open("permute/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open("README.rst") as fh:
    LONG_DESCRIPTION = fh.read()


if __name__ == "__main__":

    from setuptools import setup

    setup(
        name=DISTNAME,
        version=VERSION,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,
        python_requires=">=3.7",
        packages=["permute", "permute.tests", "permute.data", "permute.data.tests"],
        package_data={"permute.data": ["*.csv", "*/*.csv", "*/*/*.csv"]},
    )
