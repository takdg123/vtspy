from astropy import units as u

from gammapy.maps import Map

from gammapy.modeling.models import SkyModel

import gammapy.modeling.models as gammapy_model


model_dict = {
    # fermipy to gammapy
    "PowerLaw": gammapy_model.PowerLawSpectralModel,
    "LogParabola": gammapy_model.LogParabolaSpectralModel,

    # gammapy to fermipy
    "PowerLawSpectralModel": "Powerlaw",
    "LogParabolaSpectralModel": "LogParabola",
}

params = {
    "PowerLaw": {"index": float, "amplitude": float, "reference": float}
    "LogParabola": {"alpha": float, "beta": float, "amplitude": float, "reference": float}
}

def fermi_galactic_diffuse_bkg(config, norm = 1):
    diffuse_galactic_fermi = Map.read(config['model']['galdiff'][0])
    spatial_model = gammapy_model.TemplateSpatialModel(diffuse_galactic_fermi, normalize=False)
    spectral_model = gammapy_model.PowerLawNormSpectralModel()
    spectral_model.norm.value = norm
    diffuse_gald = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model,  name="galdiff")
    return diffuse_gald

def fermi_isotropic_diffuse_bkg(config, norm = 1)
    diffuse_iso = gammapy_model.create_fermi_isotropic_diffuse_model(
        filename=config['model']['isodiff'][0], 
        interp_kwargs={"fill_value": None})
    diffuse_iso.spectral_model.model2.value = norm
    return diffuse_iso

def spectral_model(src):
    gammapy_model = model_dict[src['SpectrumType']]
    gammapy_model()




    if src['SpectrumType'] == 'PowerLaw':
        factor = src['spectral_pars']['Prefactor']['scale']
        spectral_model = PowerLawSpectralModel(
            index=src['spectral_pars']['Index']['value'], 
            amplitude="{:.3e} cm-2 s-1 MeV-1".format(src['spectral_pars']['Prefactor']['value']*factor),
            reference="{:.5f} MeV".format(src['spectral_pars']['Scale']['value'])
        )
        self._re_scaling(spectral_model, src, "Prefactor")
        
    elif src['SpectrumType'] == 'LogParabola':
        factor = src['spectral_pars']['norm']['scale']
        spectral_model = LogParabolaSpectralModel(
            alpha=src['spectral_pars']['alpha']['value'], 
            beta=src['spectral_pars']['beta']['value'], 
            amplitude="{:.3e} cm-2 s-1 MeV-1".format(src['spectral_pars']['norm']['value']*factor),
            reference="{:.5f} MeV".format(src['spectral_pars']['Eb']['value'])
        )
        self._re_scaling(spectral_model, src, "norm")
        
    elif src['SpectrumType'] == 'PLSuperExpCutoff':
        factor = src['spectral_pars']['Prefactor']['scale']
        lambda_temp = 1./src['spectral_pars']['Cutoff']['value']*1e6
        spectral_model = ExpCutoffPowerLawSpectralModel(
            index=src['spectral_pars']['Index1']['value'], 
            lambda_="{:.5f} TeV-1".format(lambda_temp),
            alpha=src['spectral_pars']['Index2']['value'], 
            amplitude="{:.3e} cm-2 s-1 MeV-1".format(src['spectral_pars']['Prefactor']['value']*factor),
            reference="{:.5f} MeV".format(src['spectral_pars']['Scale']['value'])
        )
        self._re_scaling(spectral_model, src, "Prefactor")
        
    else:
        print("[Error] This type of the spectral model is not yet supported;", src['SpectrumType'])
        raise

    spectral_model = self._set_minmax(spectral_model, src)

    source = SkyModel(
        spectral_model=spectral_model,
        spatial_model=spatial_model,
        name=src.name,
    )






        if src['SpatialModel'] == "PointSource":
            spatial_model = PointSpatialModel(
            lon_0="{:.3f} deg".format(src['spatial_pars']['RA']['value']), lat_0="{:.3f} deg".format(src['spatial_pars']['DEC']['value']), frame="icrs"
            )
            spatial_model.lon_0.min = -360
            spatial_model.lon_0.max = 360
            spatial_model.lat_0.min = -90
            spatial_model.lat_0.max = 90
            
        elif src['SpatialModel'] == "RadialGaussian":
            spatial_model = GaussianSpatialModel(
            lon_0="{:.3f} deg".format(src['spatial_pars']['RA']['value']), lat_0="{:.3f} deg".format(src['spatial_pars']['DEC']['value']), sigma= "{:.3f} deg".format(src['spatial_pars']['Sigma']['value']), frame="icrs"
            )
            spatial_model.lon_0.min = -360
            spatial_model.lon_0.max = 360
            spatial_model.lat_0.min = -90
            spatial_model.lat_0.max = 90
            spatial_model.sigma.min = 0
            spatial_model.sigma.max = 180

        elif src['SpatialModel'] =="SpatialMap":
            spatial_model_temp = Map.read(src['Spatial_Filename'])
            spatial_model_temp.unit = "sr-1"
            spatial_model =  TemplateSpatialModel(spatial_model_temp, normalize=False)
        else:
            print("[Error] This type of the spatial model is not yet supported;", src['SpatialModel'])
            raise

        spatial_model.parameters.freeze_all()
        
        sources.append(source)
    return sources
