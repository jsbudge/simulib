from setuptools import setup, find_packages
from pathlib import Path
ROOT_DIR = Path(__file__).absolute().parent

setup(
   name='simulib',
   version='1.0',
   description='simulib',
   license="MIT",
   long_description='Simulation Functions for the Common Man',
   author='Jeff Budge',
   author_email='jbudge@artemisinc.net',
   url="http://www.foopackage.example/",
   packages=find_packages(),  #same as name
   install_requires=['numpy', 'scipy', 'numba', 'open3d', 'tqdm', 'plotly', 'matplotlib', 'sdrparse'], #external packages as dependencies
   package_dir={'simulib': 'simulib'},
   package_data={'simulib': ['geoids/*.DAT']},
   include_package_data=True,
)

