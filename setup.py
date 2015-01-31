#! /usr/bin/env python

descr = """Permutation tests and confidence sets for Python

"""

DISTNAME            = 'permute'
DESCRIPTION         = 'Permutation tests and confidence sets for Python'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Permute developers'
MAINTAINER_EMAIL    = 'permute@googlegroups.com'
URL                 = 'http://github.com/permute/permute'
LICENSE             = 'BSD'
DOWNLOAD_URL        = 'http://github.com/permute/permute'
VERSION             = '0.1dev'
PYTHON_VERSION      = (2, 7)

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
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],

    packages=['permute', 'permute.data', 'permute.tests'],
    package_data={ 'permute.data': ['*.csv'] }
)
