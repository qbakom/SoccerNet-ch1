from setuptools import find_packages, setup

setup(
    name='synloc',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='SoccerNet SynLoc 2026 Challenge - Single-Frame World-Coordinate Athlete Detection & Localization',
    author='qbakom',
    license='MIT',
    python_requires='>=3.11',
    entry_points={
        'console_scripts': [
            'synloc=synloc.cli:cli',
        ],
    },
)
