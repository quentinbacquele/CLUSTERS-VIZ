from setuptools import setup, find_packages

setup(
    name='CLUSTERS-VIZ',
    version='0.1.0',
    author= 'Quentin Bacquelé',
    author_email= 'quentin.bacquele@etu.unistra.fr',
    url= 'https://github.com/quentinbacquele/CLUSTERS-VIZ',
    packages=find_packages(),
    install_requires=[
        'dash',
        'dash-bootstrap-components',
        'numpy',
        'plotly',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'mfid=mfid.cli:main',
        ],
    },
)