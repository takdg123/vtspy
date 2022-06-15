Installation
============

This package is mainly based on three packages: `Gammapy
<https://gammapy.org/>`_, `Fermi Science Tools
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/>`_, and `Fermipy
<https://fermipy.readthedocs.io/en/latest/>`_. 

FermiTools
~~~~~~~~~~

We recommand to install Fermitools first. Run::

  conda create -n vtspy -c conda-forge -c fermi fermitools python=3 clhep=2.4.4.1
 
This will generate a ``conda`` environment called ``vtspy``. For details, see `Installation-Instructions <https://github.com/fermi-lat/Fermitools-conda/wiki/Installation-Instructions/>`_.

Gammapy and Fermipy
~~~~~~~~~~~~~~~~~~~

Then, install fermipy within the ``conda`` environment (``vtspy``), see also `install <https://fermipy.readthedocs.io/en/latest/install.html#install/>`_::

  conda activate vtspy
  pip install fermipy

The gammapy package is one of dependencies of the fermipy package so that you do not need to install it additionally.

vtspy
~~~~~

Run::

  git clone https://github.com/takdg123/vtspy.git
 
or

  gh repo clone takdg123/vtspy
  pip install vtspy

Or alternatively, install it from source::

  pip install --user git+https://github.com/takdg123/vtspy@master
