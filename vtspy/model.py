from astropy import units as u

from gammapy.maps import Map

from gammapy.modeling.models import SkyModel

import gammapy.modeling.models as gammapy_model

model_dict = {
    # fermipy to gammapy
    "PowerLaw": gammapy_model.PowerLawSpectralModel,
    "LogParabola": gammapy_model.LogParabolaSpectralModel,
    "PLSuperExpCutoff2":gammapy_model.ExpCutoffPowerLawSpectralModel,

    # gammapy to fermipy
    "PowerLawSpectralModel": "Powerlaw",
    "LogParabolaSpectralModel": "LogParabola",
    "ExpCutoffPowerLawSpectralModel": "PLSuperExpCutoff2"
}

params = {
    "PowerLaw": {"Index": ["index", u.dimensionless_unscaled, u.dimensionless_unscaled, -1], 
                 "Prefactor": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1], 
                 "Scale": ["reference",u.MeV, u.TeV, 1]},
    "LogParabola": {"alpha": ["alpha", u.dimensionless_unscaled, u.dimensionless_unscaled, 1], 
                    "beta": ["beta", u.dimensionless_unscaled, u.dimensionless_unscaled, 1], 
                    "norm": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1],
                    "Eb": ["reference", u.MeV, u.TeV, 1]},
    "PLSuperExpCutoff2": {"Index1": ["index", u.dimensionless_unscaled, u.dimensionless_unscaled, -1], 
                    "Index2": ["alpha", u.dimensionless_unscaled, u.dimensionless_unscaled, 1], 
                    "Expfactor": ["lambda_", 1/u.MeV, 1/u.TeV, 1], 
                    "Prefactor": ["amplitude", 1/u.cm**2/u.s/u.MeV,  1/u.cm**2/u.s/u.TeV, 1],
                    "Scale": ["reference", u.MeV, u.TeV, 1]},
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
    from fermipy.roi_model import Source
    
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

def default_model(model, **kwargs):
    if (model.lower() == "powerlaw") or (model.lower() == "pl"):
        amplitude = kwargs.pop("amplitude", 1e-12)
        index = kwargs.pop("index", 2.5)
        reference = kwargs.pop("reference", 1)

        spectral_model = gammapy_model.PowerLawSpectralModel(
            amplitude=amplitude * u.Unit("cm-2 s-1 TeV-1"),
            index=index,
            reference=reference * u.TeV,
        )
    elif model.lower() == "logparabola":
        amplitude = kwargs.pop("amplitude", 1e-12)
        alpha = kwargs.pop("alpha", 3)
        beta = kwargs.pop("beta", 2)
        reference = kwargs.pop("reference", 1)

        spectral_model = gammapy_model.LogParabolaSpectralModel(
             amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
             alpha=3,
             beta=2,
             reference=1 * u.TeV,
        )
    elif model.lower() == "agnpy":
        from .external.agnpy import agnpy_spectral_model
        
        spectral_model = agnpy_spectral_model()
        
        z = kwargs.pop("redshift", 0)
        d_L = Distance(z=z).to("cm")

        norm_e = kwargs.pop("norm_e", 5e-6)
        norm_e = norm_e/u.cm**3
        p1 = kwargs.pop("p1", 0)
        p2 = kwargs.pop("p2", 3)
        
        delta_D = kwargs.pop("delta_D", 10)
        log10_B = kwargs.pop("log10_B", 1)
        t_var = kwargs.pop("t_var", 1)
        t_var = t_var* u.d

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
        
        spectral_model.norm_e.quantity = norm_e
        spectral_model.norm_e._is_norm = True
        spectral_model.p1.quantity = p1
        spectral_model.p2.quantity = p2
        spectral_model.log10_gamma_b.quantity = log10_gamma_b
        spectral_model.log10_gamma_min.quantity = log10_gamma_min
        spectral_model.log10_gamma_min.frozen = True
        spectral_model.log10_gamma_max.quantity = log10_gamma_max
        spectral_model.log10_gamma_max.frozen = True

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

