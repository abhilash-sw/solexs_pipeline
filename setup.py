#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Abhilash Sarwade",
    author_email='sarwade@ursc.gov.in',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Data Pipeline for SoLEXS on-board Aditya-L1",
    entry_points={
        'console_scripts': [
            'solexs_pipeline=solexs_pipeline.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='solexs_pipeline',
    name='solexs_pipeline',
    packages=find_packages(include=['solexs_pipeline', 'solexs_pipeline.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    #url='https://github.com/leme-cosmo/solexs_pipeline',
    url = 'http://cdeg.isac.dos.gov.in:8081/gitlab/sarwade/solexs_pipeline',
    version='0.1.0',
    zip_safe=False,
)
