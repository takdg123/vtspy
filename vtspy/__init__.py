import numpy as np
import matplotlib.pyplot as plt

from .version import __version__

from . import utils
from .config import FermiConfig
from .download import DownloadFermiData

from astropy.visualization import astropy_mpl_style, quantity_support

plt.style.use(astropy_mpl_style)

quantity_support()
