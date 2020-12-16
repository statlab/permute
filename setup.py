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


with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()


def write_version_py(filename="permute/version.py"):
    template = """# THIS FILE IS GENERATED FROM THE PERMUTE SETUP.PY
version='%s'
"""

    try:
        fname = os.path.join(os.path.dirname(__file__), filename)
        with open(fname, "w") as f:
            f.write(template % VERSION)
    except OSError:
        raise OSError(
            "Could not open/write to permute/version.py - did you "
            "install using sudo in the past? If so, run\n"
            "sudo chown -R your_username ./*\n"
            "from package root to fix permissions, and try again."
        )


if __name__ == "__main__":

    write_version_py()

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
            "Programming Language :: Python :: 3.6",
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
        python_requires=">=3.6",
        packages=["permute", "permute.tests", "permute.data", "permute.data.tests"],
        package_data={"permute.data": ["*.csv", "*/*.csv", "*/*/*.csv"]},
    )
