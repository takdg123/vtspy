Installation
============

===========
Pre-requisite
===========

This package is mainly based on three packages: `Gammapy
<https://gammapy.org/>`_, `Fermi Science Tools
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/>`_, and `Fermipy
<https://fermipy.readthedocs.io/en/latest/>`_. We recommand to install Fermitool first.

Run::

  conda create -n fermi -c conda-forge -c fermi fermitools python=3 clhep=2.4.4.1
 
For details, see `Installation-Instructions <https://github.com/fermi-lat/Fermitools-conda/wiki/Installation-Instructions/>`_::

Then, install fermipy within the conda environment (fermi)::

  conda activate fermi
  
  pip install fermipy
  
For details, see `Installation-Instructions <https://fermipy.readthedocs.io/en/latest/install.html#install/>`_::

The gammapy package is one of dependencies of the fermipy package so that you do not need to install it separately.

===========
vtspy
===========


Run::
  
  pip install vtspy

Or alternatively, install it from source::

  pip install --user git+https://github.com/takdg123/vtspy@master
