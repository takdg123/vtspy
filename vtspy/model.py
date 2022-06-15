from astropy import units as u

from gammapy.maps import Map

from gammapy.modeling.models import SkyModel

import gammapy.modeling.models as gammapy_model

from fermipy.roi_model import Source


model_dict = {
    # fermipy to gammapy
    "PowerLaw": gammapy_model.PowerLawSpectralModel,
    "LogParabola": gammapy_model.LogParabolaSpectralModel,

    # gammapy to fermipy
    "PowerLawSpectralModel": "Powerlaw",
    "LogParabolaSpectralModel": "LogParabola",
}

params = {
    "PowerLaw": {"Index": ["index", u.dimensionless_unscaled, u.dimensionless_unscaled, -1], 
                 "Prefactor": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1], 
                 "Scale": ["reference",u.MeV, u.TeV, 1]},
    "LogParabola": {"alpha": ["alpha", u.dimensionless_unscaled, u.dimensionless_unscaled, 1], 
                    "beta": ["beta", u.dimensionless_unscaled, u.dimensionless_unscaled, 1], 
                    "norm": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1],
                    "Eb": ["reference", u.MeV, u.TeV, 1]}
}

not_spectral_shape = ["Prefactor", "norm", "Scale"]
def fermipy2gammapy(like, src):
    
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
        gpar.frozen = not(fpar.isFree())

    spatial = spatial_model(src)

    source = SkyModel(
        spatial_model=spatial,
        spectral_model=spectral,
        name=src.name,
        )
    return source

def gammapy2fermipy(spectral, src=None):
    if src is None:
        new_model = Source("new", 
                {"SpectrumType": model_dict[spectral.tag[0]]})
    else:
        new_model = Source(src.name, 
                {"ra": src.radec[0],
                 "dec": src.radec[0], 
                 "SpectrumType": model_dict[spectral.tag[0]]})

    params = {}
    for par in new_model.params.keys():
        fpar = new_model.params[par]
        gpar_setup = params[new_model['SpectrumType']][par]
        gpar = getattr(spectral, gpar_setup[0])
        val = (gpar.value*gpar_setup[2]).to(gpar_setup[1])
        params[par] = {"value":val.value, "free": not(val.frozen)}

    return params

def fermi_galactic_diffuse_bkg(config, fmodel):
    diffuse_galactic_fermi = Map.read(config['model']['galdiff'][0])
    spatial_model = gammapy_model.TemplateSpatialModel(diffuse_galactic_fermi, normalize=False)
    spectral_model = gammapy_model.PowerLawNormSpectralModel()
    spectral_model.norm.value = fmodel.params["Prefactor"]["value"]
    diffuse_gald = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model,  name="galdiff")
    return diffuse_gald

def fermi_isotropic_diffuse_bkg(config, fmodel):
    diffuse_iso = gammapy_model.create_fermi_isotropic_diffuse_model(
        filename=config['model']['isodiff'][0], 
        interp_kwargs={"fill_value": None})
    diffuse_iso.spectral_model.model2.value = fmodel.params["Normalization"]["value"]
    diffuse_iso._name = "isodiff"
    return diffuse_iso

def gammapy_default_model(model):
    if model.lower() == "powerlaw":
        spectral_model = gammapy_model.PowerLawSpectralModel(
            amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
            index=2.5,
            reference=1 * u.TeV,
        )
    elif model.lower() == "logparabola":
        spectral_model = gammapy_model.LogParabolaSpectralModel(
             alpha=3,
             amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
             reference=1 * u.TeV,
             beta=2,
        )
    else:
        spectral_model = None
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

    spatial_model.parameters.freeze_all()
    return spatial_model

