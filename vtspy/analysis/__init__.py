from .veritas_analysis import VeritasAnalysis
import logging
from .fermi_analysis import FermiAnalysis
from .joint_analysis import JointAnalysis

logging.warning("Fermitools is not installed. Any Fermi-LAT related analysis cannot be performed.")