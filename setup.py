from setuptools import setup

setup(
    name='multiorbit',
    version='0.1.dev',
    packages=['multiorbit'],
    url='https://github.com/mwbub/multiorbit',
    author='Mathew Bub',
    author_email='mathew.bub@gmail.com',
    description='Package for running large orbit integration suites in galpy.',
    install_requires=['galpy', 'astropy']
)
