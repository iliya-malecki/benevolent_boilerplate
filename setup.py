from setuptools import setup, find_packages

from benevolent_boilerplate import __version__

setup(
    name='benevolent_boilerplate',
    version=__version__,

    url='https://github.com/iliya-malecki/benevolent_boilerplate',
    author='Iliya Malecki',
    author_email='iliyamalecki@gmail.com',
    
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'plotly == 5.9.0',
        'pandas >= 1.0',
        'numpy >= 1.20',
        'scikit-learn >= 1.0',
    ],
)