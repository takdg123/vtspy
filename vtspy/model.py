from astropy import units as u

from gammapy.maps import Map

from gammapy.modeling.models import SkyModel

import gammapy.modeling.models as gammapy_model

from astropy.coordinates import Distance

import numpy as np

from gammapy.modeling.models import EBLAbsorptionNormSpectralModel

model_dict = {
    # fermipy to gammapy
    "PowerLaw": gammapy_model.PowerLawSpectralModel,
    "LogParabola": gammapy_model.LogParabolaSpectralModel,
    "PLSuperExpCutoff4":gammapy_model.SuperExpCutoffPowerLaw4FGLDR3SpectralModel,

    # gammapy to fermipy
    "PowerLawSpectralModel": "Powerlaw",
    "LogParabolaSpectralModel": "LogParabola",
    "SuperExpCutoffPowerLaw4FGLDR3SpectralModel": "PLSuperExpCutoff4",

}

params = {
    "PowerLaw": {"Index": ["index", u.dimensionless_unscaled, u.dimensionless_unscaled, -1],
                 "Prefactor": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1],
                 "Scale": ["reference",u.MeV, u.TeV, 1]},
    "LogParabola": {"alpha": ["alpha", u.dimensionless_unscaled, u.dimensionless_unscaled, 1],
                    "beta": ["beta", u.dimensionless_unscaled, u.dimensionless_unscaled, 1],
                    "norm": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1],
                    "Eb": ["reference", u.MeV, u.TeV, 1]},
    "PLSuperExpCutoff4": {"IndexS": ["index_1", u.dimensionless_unscaled, u.dimensionless_unscaled, -1],
                    "Index2": ["index_2", u.dimensionless_unscaled, u.dimensionless_unscaled, 1],
                    "ExpfactorS": ["expfactor", u.dimensionless_unscaled, u.dimensionless_unscaled, 1],
                    "Prefactor": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1],
                    "Scale": ["reference", u.MeV, u.TeV, 1]},
}

not_spectral_shape = ["Prefactor", "norm", "Scale"]

def fermipy2gammapy(like, src, fix_pars=False):

    spectral = model_dict[src['SpectrumType']]()
    for par in src.spectral_pars.keys():
        idx = like.par_index(src.name, par)
        fpar = like.model[idx]

        idx = like.par_index(src.name, par)
        fpar = like.model[idx]
        gpar_setup = params[src['SpectrumType']][par]
        gpar = getattr(spectral, gpar_setup[0])
        val = (fpar.getValue()*fpar.getScale()*gpar_setup[1]*gpar_setup[3]).to(gpar_setup[2])
        gpar.value = val.value
        if gpar_setup[0] == "amplitude":
            gpar.min = ((1e-5*fpar.getScale()*gpar_setup[1]*gpar_setup[3]).to(gpar_setup[2])).value
            gpar.max = ((1e3*fpar.getScale()*gpar_setup[1]*gpar_setup[3]).to(gpar_setup[2])).value
        else:
            gpar.min = src["spectral_pars"][par]["min"]
            gpar.max = src["spectral_pars"][par]["max"]

        if fix_pars:
            gpar.frozen = True
        else:
            gpar.frozen = not(fpar.isFree())

    spatial = spatial_model(src)

    source = SkyModel(
        spatial_model=spatial,
        spectral_model=spectral,
        name=src.name,
        )

    return source

# def gammapy2fermipy(spectral, src=None):
#     from fermipy.roi_model import Source

#     if src is None:
#         new_model = Source("new",
#                 {"SpectrumType": model_dict[spectral.tag[0]]})
#     else:
#         new_model = Source(src.name,
#                 {"ra": src.radec[0],
#                  "dec": src.radec[0],
#                  "SpectrumType": model_dict[spectral.tag[0]]})

#     params = {}
#     for par in new_model.params.keys():
#         fpar = new_model.params[par]
#         gpar_setup = params[new_model['SpectrumType']][par]
#         gpar = getattr(spectral, gpar_setup[0])
#         val = (gpar.value*gpar_setup[2]).to(gpar_setup[1])
#         params[par] = {"value":val.value, "free": not(val.frozen)}

#     return params

def fermi_galactic_diffuse_bkg(config, fmodel, fix_pars = False):
    diffuse_galactic_fermi = Map.read(config['model']['galdiff'][0])
    spatial_model = gammapy_model.TemplateSpatialModel(diffuse_galactic_fermi, normalize=False)
    spectral_model = gammapy_model.PowerLawNormSpectralModel()
    spectral_model.norm.value = fmodel.params["Prefactor"]["value"]
    spectral_model.norm.min = 0.1
    spectral_model.norm.max = 10
    spectral_model.norm.frozen = fix_pars
    diffuse_gald = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model,  name="galdiff")
    return diffuse_gald

def fermi_isotropic_diffuse_bkg(config, fmodel, fix_pars = False):
    diffuse_iso = gammapy_model.create_fermi_isotropic_diffuse_model(
        filename=config['model']['isodiff'][0],
        interp_kwargs={"fill_value": None})
    diffuse_iso.spectral_model.model2.norm.value = fmodel.params["Normalization"]["value"]
    diffuse_iso.spectral_model.model2.norm.min = 0.1
    diffuse_iso.spectral_model.model2.norm.max = 10
    diffuse_iso.spectral_model.model2.norm.frozen = fix_pars
    diffuse_iso._name = "isodiff"

    return diffuse_iso

def default_model(model, correct_ebl=False, ebl_model="dominguez", **kwargs):
    z = kwargs.pop("z", 0)
    if (model.lower() == "powerlaw") or (model.lower() == "pl"):
        amplitude = kwargs.pop("amplitude", 1e-12* u.Unit("cm-2 s-1 TeV-1"),)
        index = kwargs.pop("index", 2.5)
        reference = kwargs.pop("reference", 1 * u.TeV)

        spectral_model = gammapy_model.PowerLawSpectralModel(
            amplitude=amplitude,
            index=index,
            reference=reference,
        )
    elif model.lower() == "logparabola":
        amplitude = kwargs.pop("amplitude", 1e-12 * u.Unit("cm-2 s-1 TeV-1"))
        alpha = kwargs.pop("alpha", 3)
        beta = kwargs.pop("beta", 2)
        reference = kwargs.pop("reference", 1 * u.TeV)

        spectral_model = gammapy_model.LogParabolaSpectralModel(
             amplitude=amplitude,
             alpha=alpha,
             beta=beta,
             reference=reference,
        )
    elif model.lower() == "agnpy":
        from .external.agnpy import agnpy_spectral_model

        spectral_model = agnpy_spectral_model()

        t_var = kwargs.pop("t_var", 0)

        if (z == 0) or (t_var == 0):
            raise
        d_L = Distance(z=z).to("cm")

        log10_norm_e = kwargs.pop("log10_norm_e", -5)
        p = kwargs.pop("p", 1)
        dp = kwargs.pop("dp", 2)

        delta_D = kwargs.pop("delta_D", 10)
        log10_B = kwargs.pop("log10_B", 0.1)

        log10_gamma_b = kwargs.pop("log10_gamma_b", 4)
        log10_gamma_min = kwargs.pop("log10_gamma_min", np.log10(500))
        log10_gamma_max = kwargs.pop("log10_gamma_max", np.log10(1e6))

        spectral_model.z.quantity = z
        spectral_model.z.frozen = True
        spectral_model.d_L.quantity = d_L
        spectral_model.d_L.frozen = True

        spectral_model.delta_D.quantity = delta_D
        spectral_model.log10_B.quantity = log10_B
        spectral_model.t_var.quantity = t_var
        spectral_model.t_var.frozen = True

        spectral_model.log10_norm_e.quantity = log10_norm_e
        spectral_model.log10_norm_e._is_norm = True
        spectral_model.p.quantity = p
        spectral_model.dp.quantity = dp
        spectral_model.log10_gamma_b.quantity = log10_gamma_b
        spectral_model.log10_gamma_min.quantity = log10_gamma_min
        spectral_model.log10_gamma_min.frozen = False
        spectral_model.log10_gamma_max.quantity = log10_gamma_max
        spectral_model.log10_gamma_max.frozen = False

    else:
        spectral_model = None
        
    if correct_ebl == True:
        ebl_absorption = EBLAbsorptionNormSpectralModel.read_builtin(ebl_model, redshift=z)
        spectral_model = spectral_model * ebl_absorption
    
    return spectral_model

def spatial_model(src):
    if src['SpatialModel'] == "PointSource":
        spatial_model = gammapy_model.PointSpatialModel(
        lon_0="{:.3f} deg".format(src['spatial_pars']['RA']['value']), lat_0="{:.3f} deg".format(src['spatial_pars']['DEC']['value']), frame="icrs"
        )

    elif src['SpatialModel'] == "RadialGaussian":
        spatial_model = gammapy_model.GaussianSpatialModel(
        lon_0="{:.3f} deg".format(src['spatial_pars']['RA']['value']), lat_0="{:.3f} deg".format(src['spatial_pars']['DEC']['value']), sigma= "{:.3f} deg".format(src['spatial_pars']['Sigma']['value']), frame="icrs"
        )
    elif src['SpatialModel'] =="SpatialMap":
        spatial_model_temp = Map.read(src['Spatial_Filename'])
        spatial_model_temp.unit = "sr-1"
        spatial_model =  gammapy_model.TemplateSpatialModel(spatial_model_temp, normalize=False)
    else:
        self._logging.error(f"This type of the spatial model is not yet supported; {src['SpatialModel']}")
        raise

    spatial_model.lon_0.min = -360
    spatial_model.lon_0.max = 360
    spatial_model.lat_0.min = -90
    spatial_model.lat_0.max = 90

    spatial_model.parameters.freeze_all()
    return spatial_model
