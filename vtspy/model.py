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
    "PowerLaw": {"Index": ["index", u.dimensionless_unscaled, u.dimensionless_unscaled], 
                 "Prefactor": ["amplitude", u.cm**2/u.s/u.MeV,  u.cm**2/u.s/u.TeV], 
                 "Scale": ["reference",u.MeV, u.TeV]},
    "LogParabola": {"alpha": ["alpha", u.dimensionless_unscaled, u.dimensionless_unscaled], 
                    "beta": ["beta", u.dimensionless_unscaled, u.dimensionless_unscaled], 
                    "norm": ["amplitude", u.cm**2/u.s/u.MeV,  u.cm**2/u.s/u.TeV],
                    "Eb": ["reference", u.MeV, u.TeV]}
}


def fermipy2gammapy(src):
    spectral = model_dict[src['SpectrumType']]()
    for par in src.params.keys():
        fpar = src.spectral_pars[par]
        gpar_setup = params[src['SpectrumType']][par]
        gpar = getattr(spectral, gpar_setup[0])
        val = (fpar["value"]*gpar_setup[1]).to(gpar_setup[2])
        gpar.value = val.value

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
        params[par] = {"value":val.value}

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
    return diffuse_iso

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
