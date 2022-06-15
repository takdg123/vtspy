import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.coordinates import Distance

from agnpy.spectra import BrokenPowerLaw
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton

from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    SpectralModel,
    SPECTRAL_MODEL_REGISTRY,
)


class agnpy_spectral_model(SpectralModel):
    """
    Wrapper of agnpy's synchrotron and SSC classes. A broken power law is assumed 
    for the electron spectrum. To limit the span of the parameters space, we fit 
    the log10 of the parameters whose range is expected to cover several orders 
    of magnitudes (normalisation, gammas, size and magnetic field of the blob).
    """

    tag = ["agnpy(SYN+SSC)", "agnpy"]
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=10)
    p1 = Parameter("p1", 2.1, min=-2.0, max=5.0)
    p2 = Parameter("p2", 3.1, min=-2.0, max=5.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=7)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=4, max=8)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=0, max=60)
    log10_B = Parameter("log10_B", -1, min=-4, max=2)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)

    @staticmethod
    def evaluate(
        energy,
        log10_k_e,
        p1,
        p2,
        log10_gamma_b,
        log10_gamma_min,
        log10_gamma_max,
        z,
        d_L,
        delta_D,
        log10_B,
        t_var,
    ):
        # conversions
        print(energy)
        
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")
        pars = (z, d_L, delta_D, B, R_b, BrokenPowerLaw, k_e,
                p1, p2, gamma_b, gamma_min, gamma_max)
        pars = (z, d_L, delta_D, B, R_b, BrokenPowerLaw, k_e,
                p1, p2, gamma_b, gamma_min, gamma_max)
        nu = energy.to("Hz", equivalencies=u.spectral())
        dim = len(energy.shape)
        
        if dim == 2:
            sed_synch_l = Synchrotron.evaluate_sed_flux(nu[:,0],*pars)
            sed_synch_h = Synchrotron.evaluate_sed_flux(nu[:,1],*pars)
            sed_synch = np.asarray([sed_synch_l,sed_synch_h]).T
            
            sed_ssc_l = SynchrotronSelfCompton.evaluate_sed_flux(nu[:,0],*pars)
            sed_ssc_h = SynchrotronSelfCompton.evaluate_sed_flux(nu[:,1],*pars)
            sed_ssc = np.asarray([sed_ssc_l,sed_ssc_h]).T
            
            sed_units = u.erg / u.cm / u.cm / u.second
            sed = (sed_synch + sed_ssc) * sed_units
        else:
            sed_synch = Synchrotron.evaluate_sed_flux(nu,*pars)
            sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(nu,*pars)
            sed = (sed_synch + sed_ssc) 
        return (sed / energy ** 2).to("1 / (cm2 TeV s)")


