#! /usr/bin/env python

descr = """Permutation tests and confidence sets for Python

"""

DISTNAME = 'permute'
DESCRIPTION = 'Permutation tests and confidence sets for Python'
LONG_DESCRIPTION = descr
MAINTAINER = 'Permute developers'
MAINTAINER_EMAIL = 'permute@googlegroups.com'
URL = 'http://statlab.github.io/permute/'
LICENSE = 'BSD License'
DOWNLOAD_URL = 'http://github.com/statlab/permute'
VERSION = '0.1.alpha2'
PYTHON_VERSION = (2, 7)

INSTALL_REQUIRES = [
    'numpy',
    'scipy'
]

TESTS_REQUIRE = [
    'coverage',
    'nose',
    'flake8'
]

import os


def write_version_py(filename='permute/version.py'):
    template = """# THIS FILE IS GENERATED FROM THE PERMUTE SETUP.PY
version='%s'
"""

    try:
        fname = os.path.join(os.path.dirname(__file__), filename)
        with open(fname, 'w') as f:
            f.write(template % VERSION)
    except IOError:
        raise IOError("Could not open/write to permute/version.py - did you "
                      "install using sudo in the past? If so, run\n"
                      "sudo chown -R your_username ./*\n"
                      "from package root to fix permissions, and try again.")


if __name__ == "__main__":

    write_version_py()

    from setuptools import setup
    setup(
        name=DISTNAME,
        version=VERSION,
        license=LICENSE,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,

        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],

        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE,

        packages=['permute', 'permute.data', 'permute.tests', 'permute.data.tests'],
        package_data={'permute.data': ['*.csv', '*/*.csv', '*/*/*.csv']}
    )
