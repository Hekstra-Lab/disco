from setuptools import setup, find_packages

setup(
    name='disco',
    version='0.0.1',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    include_package_data=True,
    package_data={'' : ['data/pdb_data.csv.bz2']},
    description='Compute random laue spot centroids for training machine learning models',
    install_requires=[
        "reciprocalspaceship>=0.9.15",
        "matplotlib",
        "celluloid",
    ],
    entry_points = {
        "console_scripts": [
            "disco.wow=disco.commandline.wow:main",
        ]
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)
