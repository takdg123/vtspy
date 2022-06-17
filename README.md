This is the ``vtspy`` documentation page. ``vtspy`` is a python package that facilitates analysis of data from `VERITAS (Very Energetic Radiation Imaging Telescope Array System) <https://veritas.sao.arizona.edu/>`_.
The vtspy package is built on two python packages, `gammapy
<https://gammapy.org/>`_. and `fermipy
<https://fermipy.readthedocs.io/en/latest/>`_, and provides a set of high-level tools for performing various VERITAS analyses
* Generate a configuration file based on information obtained from a DL3 fit file (``V2DL3`` convertor), which can be used in ``gammapy`` and ``fermipy``. 

* Perform spectral and temporal analyses with ``gammapy``.

* Prepare Fermi-LAT data and perform analysis for the same day of the VERITAS observation with ``fermipy``.

* Prepare the VERITAS and Fermi-LAT datasets for the ``gammapy`` analysis and perform a joint-fit analysis with VERITAS and Fermi-LAT datasets