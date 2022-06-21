from setuptools import setup, find_packages
from pathlib import Path


## read __version__
with open(Path(__file__).parent.absolute().joinpath('vtspy/version.py')) as f:
    exec(f.read())


setup(name='vtspy',
      version=__version__,
      description="VERITAS analysis tool",
      packages=find_packages(),
      install_requires=['astropy',
                        'numpy',
                        'matplotlib>=3.5.2',
                        'gammapy',
                        'uproot',
                        'html2text'
                        ],
      extras_require={'tests': ['pytest', 'pytest-ordering'],
                      'examples': ['ipywidgets', 'ipympl', 'nodejs']
                      },
      package_data={
        'vtspy': ['refdata/*.fit', 'refdata/*.dat']
      },
      include_package_data=True,
      author='Donggeun Tak',
      author_email='donggeun.tak@gmail.com',
      url='https://vtspy.readthedocs.io/en/latest/',
      long_description="",
      use_scm_version={
          "write_to": Path(__file__).parent.joinpath("vtspy/version.py"),
          "write_to_template": "__version__ = '{version}'",
      },
      )
