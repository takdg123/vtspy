#from .fermi_analysis.download import DownloadFermiData
#from .fermi_analysis.config import config
#from .fermi_analysis.analysis import FermiAnalysis

#from .gamma_analysis.analysis import GammaAnalysis

#from .utils import *

import numpy as np
import matplotlib.pyplot as plt

from .version import __version__

from . import utils
from . import config
from . import download

from pathlib import Path
SCRIPT_DIR = str(Path(__file__).parent.absolute())

from astropy.visualization import astropy_mpl_style, quantity_support

plt.style.use(astropy_mpl_style)

quantity_support()
