from setuptools import setup, find_packages

setup(
    name='protein_ddpm',
    version='1.0.0',
    url='https://github.com/RossDeVito/protein_ddpm.git',
    author='Ross DeVito',
    description='Conditional protein string generation.',
    packages=find_packages(),    
    # install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)