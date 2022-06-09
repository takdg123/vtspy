import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from .version import __version__

from . import utils
from .config import GammaConfig
from .download import DownloadFermiData

from astropy.visualization import astropy_mpl_style, quantity_support

plt.style.use(astropy_mpl_style)

quantity_support()
