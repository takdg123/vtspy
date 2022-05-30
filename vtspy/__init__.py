from .fermi_analysis.download import DownloadFermiData
from .fermi_analysis.config import config
from .fermi_analysis.fermi_analysis import FermiAnalysis

from .gamma_analysis.gamma_analysis import GammaAnalysis

from .utils import *

import numpy as np
import matplotlib.pyplot as plt

from .version import __version__

plt.style.use(astropy_mpl_style)

quantity_support()

