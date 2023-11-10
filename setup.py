from setuptools import setup 

setup( 
    name='group-code', 
    version='0.1', 
    description='', 
    author='Michael Rust', 
    author_email='mrust2@vols.utk.edu', 
    packages=['group-code'], 
    install_requires=[ 
        'numpy', 
        'pandas',
        'pyvisa', 
    ], 
) 