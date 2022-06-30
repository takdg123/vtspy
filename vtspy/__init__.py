import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from .version import __version__

from . import utils
from .config import JointConfig
from .download import DownloadFermiData
from .analysis import *
from .model import default_model
from gammapy.modeling import Fit
gammapy_default_fit = Fit()

from astropy.visualization import astropy_mpl_style, quantity_support

plt.style.use(astropy_mpl_style)

quantity_support()
