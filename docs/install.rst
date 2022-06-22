Installation
============

This package is mainly based on three packages: `Gammapy
<https://gammapy.org/>`_, `Fermi Science Tools
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/>`_, and `Fermipy
<https://fermipy.readthedocs.io/en/latest/>`_. 

FermiTools
~~~~~~~~~~

We recommand to install Fermitools first with ``mamba``. Run::

  conda install mamba -n base -c conda-forge
  
then::

  mamba create --name vtspy -c conda-forge -c fermi -c fermi/label/rc python=3.9 "fermitools>=2.1.0" healpy gammapy
 
This will generate a ``mamba`` environment called ``vtspy``. For details, see `Installation-Instructions <https://github.com/fermi-lat/Fermitools-conda/wiki/Installation-Instructions/>`_.

Gammapy and Fermipy
~~~~~~~~~~~~~~~~~~~

Then, install ``fermipy`` within the ``conda`` environment (``vtspy``), see also `install <https://fermipy.readthedocs.io/en/latest/install.html#install/>`_::

  mamba activate vtspy
  pip install fermipy

The ``gammapy`` package is one of dependencies of the ``fermipy`` package so that you do not need to install it additionally.

vtspy
~~~~~

Run::
  
  pip install vtspy

or::
  
  git clone https://github.com/takdg123/vtspy.git
  cd vtspy
  pip install .
