
import os
from setuptools import find_packages, setup
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='colav',
    version='0.1', 
    author='Ammaar A. Saeed', 
    author_email='aasaeed@college.harvard.edu',
    description=(
    '''Package to explore the conformational landscape of a protein using existing
    experimental structures'''), 
    license='', 
    url='',
    long_description=read('README.md'),
    packages=find_packages(),
)