import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.patches as Patches

from astropy import units as u
from astropy.coordinates import SkyCoord

from fermipy.gtanalysis import GTAnalysis

from gammapy.data import EventList
from gammapy.datasets import Datasets, MapDataset

from gammapy.irf import EnergyDependentTablePSF, PSFMap, EDispMap
from gammapy.maps import Map, MapAxis, WcsGeom

from gammapy.utils.scripts import make_path

from gammapy.modeling.models import (
    ConstantSpatialModel,
    PointSpatialModel,
    GaussianSpatialModel,
    TemplateSpatialModel,
    SkyModel,
    Models,
)

from gammapy.modeling.models import (
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    LogParabolaSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    TemplateSpectralModel,    
)

from gammapy.modeling import Fit

import fermipy.wcs_utils as wcs_utils
import fermipy.utils as fermi_utils

from regions import CircleSkyRegion
from pathlib import Path

from fermipy.roi_model import Source


paramCovf2g = {'norm': ['amplitude', 1],
            'Prefactor': ['amplitude', 1],
            'Eb': ['reference', 1e-6],
            'Scale': ['reference', 1e-6],
            'Index1': ['index', 1],
            'Index2': ['alpha', 1],
            'Cutoff': ['lambda_', 1],
            }

fPLSuperExpCutoff = {'SpectrumType' : 'PLSuperExpCutoff', 
                    'Prefactor' : 1, 
                    'Index1': -1.7, 
                    'Scale' : 200, 
                    'Cutoff': 3000, 
                    'Index2': 1.5}

def calc_scale(factor):
    if int(np.log10(factor)) >=0:
        scale = 10**(int(np.log10(factor)))
    else:    
        scale = 10**(int(np.log10(factor))-1)
    return scale
class FermiUtils(GTAnalysis):
    def __init__(self, file='config_fermi.yaml', roi = "initial",  verbosity=True, overwrite=False, **kwargs):
        
        self._verbosity = verbosity
        self._fermi_energy_bin = np.logspace(2, 5.5, 8)* u.MeV

        if not(os.path.isdir("./results")):
            os.system("mkdir results")

        self.fermi_data = {}
        
        self._fermi_outdir = Path("fermi")

        super().__init__(file, logging={'verbosity' : 1}, **kwargs)
        
        if not(os.path.isfile(self._fermi_outdir / "initial.fits")):
                print("[Error] The Fermi-LAT data reduction and configuration have not been done. ")
                FermiAnalysis(file=file, roi=roi, verbosity=verbosity)
        
        if self.verbosity:
            print("[Log][0/6] Initializing the Fermi-LAT setup...")
            print("[Log][1/6] Loading fermipy setup...")

        self.load_roi(roi)

        self._target = self.roi.sources[0]
        self._target_id = 0
        try:
            self.output = np.load(self._fermi_outdir / "output.npy", allow_pickle=True).item()
        except:
            pass

        if self.verbosity:
            print("[Log][2/6] Loading the Fermi-LAT events...")
        self._load_fermi_events()

        if self.verbosity:
            print("[Log][3/6] Loading the Fermi-LAT IRFs...")
        self._load_fermi_irfs()            

        if self.verbosity:
            print("[Log][4/6] Importing the list of sources from fermipy...")

        self._fermipy_2_gammpy_models()
        self._sources = self._set_fermi_sources(self.roi)
        
        if self.verbosity:
            print("[Log][5/6] Importing the list of diffuse backgrounds from fermipy...")
        self._diff_bkg = self._set_fermi_diffuse_bkg(self.roi)
        self._models = Models(self._sources+self._diff_bkg)

        if self.verbosity:
            print("[Log][6/6] Finalizing the Fermi-LAT setup", end='\r')

        self.FermiDatasets = self.setFermiDatasets()

        if overwrite:
            self.FermiDatasets.write(self._fermi_outdir / "fermi_datasets.yaml", overwrite=True)

        if self.verbosity:
            print("[Log][6/6] Completed.                             ")

        
    def setFermiDatasets(self):
        datasets = Datasets()
        datasets.append(MapDataset(
                name="Fermi-LAT",
                models=self.models, 
                counts=self.fermi_data['counts'], 
                exposure=self.fermi_data['exposure'], 
                psf=self.fermi_data['psf'], 
                edisp=self.fermi_data['edisp'],

            ))
        return datasets

    def freeSources(self, min_ts = None, ts_cut=0.01, target_free=True, **kwargs):
        if min_ts != None:
            kwargs = {**kwargs, "minmax_ts":[None,min_ts], "free":False}

        self.free_sources(**kwargs)
        
        if target_free:
            self.freeSource(self.target.name)

        self._update_models(ts_cut=ts_cut)

    def freeSource(self, name, ts_cut = 0.01, **kwargs):
        self.free_source(name, **kwargs)
        self._update_models(ts_cut=ts_cut)        

    def _update_models(self, bkg=False, ts_cut = 0.01):
        sources = self._set_fermi_sources(self.roi, ts_cut = ts_cut)
        if bkg:
            self._diff_bkg = self._set_fermi_diffuse_bkg(self.roi)
        self._models = Models(sources+self._diff_bkg)        

    def _load_fermi_events(self):
        events = EventList.read(self._fermi_outdir / "ft1_00.fits")
        src_pos = SkyCoord(self.config['selection']['glon'], self.config['selection']['glat'], unit="deg", frame="galactic")
        energy_axis = MapAxis.from_edges(self._fermi_energy_bin, name="energy", unit="MeV", interp="log")
        counts = Map.create(skydir=src_pos, width=self.config['binning']['roiwidth'], proj=self.config['binning']['proj'], binsz=self.config['binning']['binsz'], frame='galactic', axes=[energy_axis], dtype=float,)
        counts.fill_by_coord({"skycoord": events.radec, "energy": events.energy})
        self.fermi_data['counts'] = counts


    def _load_fermi_irfs(self):
        # Exposure
        counts = self.fermi_data['counts']

        expmap = Map.read(self._fermi_outdir / "bexpmap_00.fits")
        axis = MapAxis.from_nodes(
            counts.geom.axes[0].center, 
            name="energy_true",
            unit="MeV", 
            interp="log"
        )

        geom = WcsGeom(wcs=counts.geom.wcs, npix=counts.geom.npix, axes=[axis])
        exposure = expmap.interp_to_geom(geom)
        self.fermi_data['exposure'] = exposure
        
        # PSF
        psf_table = EnergyDependentTablePSF.read(self._fermi_outdir / "gtpsf_00.fits")
        psf = PSFMap.from_energy_dependent_table_psf(psf_table)
        self.fermi_data['psf'] = psf

        # Energy dispersion
        e_true = exposure.geom.axes["energy_true"]
        edisp = EDispMap.from_diagonal_response(energy_axis_true=e_true)
        self.fermi_data['edisp'] = edisp

    def _fermipy_2_gammpy_models(self):
        for src in self.roi.sources:
            if src['SpectrumType'] =='PLSuperExpCutoff2':
                alt_src = Source(src.name, {'ra': src.spatial_pars['RA']['value'], 'dec': src.spatial_pars['DEC']['value'], 
                                            **fPLSuperExpCutoff})
                
                for par in src.spectral_pars.keys():
                    if par == 'Expfactor':
                        alt_src.spectral_pars['Cutoff']['max'] = 5000
                        alt_src.spectral_pars['Cutoff']['value'] = (src.spectral_pars['Expfactor']['value']*src.spectral_pars['Expfactor']['scale'])**(-1./src.spectral_pars['Index2']['value'])
                        alt_src.spectral_pars['Cutoff']['free'] = True
                    elif par == 'Prefactor':
                        alt_src.spectral_pars[par]['scale'] = 1e-7
                        alt_src.spectral_pars[par]['value'] = (src.spectral_pars[par]['value']*src.spectral_pars[par]['scale']/alt_src.spectral_pars[par]['scale'])
                    else:
                        alt_src.spectral_pars[par]['value'] = src.spectral_pars[par]['value']

                    if par !="Scale" and par !='Expfactor':
                        alt_src.spectral_pars[par]['free'] = True

                ts_val = src['ts']
                names = src.associations
                self.delete_source(src.name)
                self.add_source(alt_src.name, alt_src)
                self.roi[alt_src.name]['ts'] = ts_val

                for name in names:
                    self.roi[alt_src.name].add_name(name)

    def _gammapy_2_fermipy_models(self, save=False):
        for model in self.models:
            name = model.name

            try:
                src = self.roi[name]
            except:
                print("[Error] {} is not in the fermipy model list.".format(name))
                continue

            if name == 'isodiff':
                idx = self.like.par_index('isodiff', 'Normalization')
                norm = getattr(model.spectral_model.model2, 'norm')
                self.like[idx].setValue(norm.value)
            elif name == 'galdiff':
                idx = self.like.par_index('galdiff', 'Prefactor')
                norm = getattr(model.spectral_model, 'amplitude')
                self.like[idx].setValue(norm.value/1e6)

                idx = self.like.par_index('galdiff', 'Index')
                index = getattr(model.spectral_model, 'index')
                self.like[idx].setValue(index.value)
            else:
                for par in self.roi[name].params.keys():
                    try:
                        alt_par = paramCovf2g[par][0]
                        factor = paramCovf2g[par][1]
                    except:
                        alt_par = par.lower()
                        factor = 1
                    temp = getattr(model.spectral_model, alt_par)
                    
                    idx = self.like.par_index(name, par)
                    if par != 'Prefactor' and par != "norm":
                        orig_val = self.like[idx].getValue()
                        
                        if par == 'Cutoff':
                            new_val = 1./temp.value*1e6
                            new_err = 1./temp.error*1e6
                        else:
                            new_val = temp.value/factor
                            new_err = temp.error/factor
                        
                        if abs(orig_val/new_val-1)>1e-3:
                            self.like[idx].setValue(new_val) 
                            self.like[idx].setError(new_err) 
                    elif par == "Prefactor" or par == "norm":
                        orig_val = self.like[idx].getValue()
                        scale = self.roi[name]['spectral_pars'][par]['scale']
                        new_val = temp.value*1e-6/scale
                        new_err = temp.error*1e-6/scale
                        self.like[idx].setValue(new_val)
                        self.like[idx].setError(new_err)
                                                    
            self._sync_params(name)
            self.update_source(name)

        if save: self.write_xml("gamma")

    def _default_fermi_setup(self):
        
        self.freeSources(free=False, loglevel=0)
        self.freeSources(free=True, pars='norm', loglevel=0)
        self.freeSources(free=True, minmax_npred=[100, None], loglevel=0)

    def _source_analysis_setup(self):
        cond1 = self.get_free_param_vector()
        self.freeSources(free=False)
        self.freeSources(skydir=self.target.skydir, distance=3.0, free=True, pars="norm")
        cond2 = self.get_free_param_vector()
        cond = np.asarray(cond1)+np.asarray(cond2)
        self.set_free_param_vector(list(cond))
        self._update_models()
        
    def _set_fermi_sources(self, roi_data, ts_cut = 0.01, roi_cut = 3):
        # 4FGL sources
        sources = []
        for src in roi_data.get_sources():

            if (src.name=='isodiff') or (src.name=='galdiff') or np.isnan(src['ts']):
                continue

            if ((src.name != self.target.name) 
                and (src['offset'] > roi_cut) 
                and (src['ts'] < ts_cut)):
                continue

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


            sources.append(source)
        return sources

    def _re_scaling(self, spectral_model, src, par):
        val = src['spectral_pars'][par]['value']
        factor = spectral_model.amplitude.factor
        scale=calc_scale(factor)
        val_scale = max(calc_scale(val), 10**-3)
        tot_scale = scale/val_scale
        
        spectral_model.amplitude.factor = factor/tot_scale
        spectral_model.amplitude.scale = tot_scale

    def _set_fermi_diffuse_bkg(self, roi_data):
        # Galactic diffuse
        diffuse_galactic_fermi = Map.read(self.config['model']['galdiff'][0])
        diffuse_galactic_fermi.unit = "sr-1"
        spatial_model = TemplateSpatialModel(diffuse_galactic_fermi, normalize=False)
        spectral_model = PowerLawSpectralModel(
                    index=roi_data['galdiff']['spectral_pars']['Index']['value'], 
                    amplitude="{:.3e} cm-2 s-1 MeV-1".format(roi_data['galdiff']['spectral_pars']['Prefactor']['value']),
                    reference="{:.0f} MeV".format(roi_data['galdiff']['spectral_pars']['Scale']['value'])
                )
        factor = spectral_model.amplitude.factor
        scale = calc_scale(factor)

        spectral_model.amplitude.factor = factor/scale
        spectral_model.amplitude.scale = scale
        spectral_model.amplitude.min = 0.1*scale
        spectral_model.amplitude.max = 10*scale
        spectral_model.index.min = -1
        spectral_model.index.max = 1
        spectral_model.reference.frozen = True

        diffuse_iem = SkyModel(spectral_model=spectral_model, spatial_model=spatial_model,  name="galdiff")

        # Isotropic diffuse
        diffuse_isotropic_fermi = np.loadtxt(make_path(self.config['model']['isodiff'][0]))
        energy = u.Quantity(diffuse_isotropic_fermi[:, 0], "MeV", copy=False)
        values = u.Quantity(diffuse_isotropic_fermi[:, 1], "MeV-1 s-1 cm-2", copy=False)

        spatial_model = ConstantSpatialModel()
        spatial_model.value.min = 1
        spatial_model.value.max = 1
        
        spectral_model = (TemplateSpectralModel(energy=energy, values=values) * PowerLawNormSpectralModel())
        spectral_model.model2.tilt.min = 1
        spectral_model.model2.tilt.max = 1
        spectral_model.model2.norm.value = roi_data['isodiff']['spectral_pars']['Normalization']['value']
        spectral_model.model2.norm.min = roi_data['isodiff']['spectral_pars']['Normalization']['min']
        spectral_model.model2.norm.max = roi_data['isodiff']['spectral_pars']['Normalization']['max']
        spectral_model.model2.reference.min = 1
        spectral_model.model2.reference.max = 1

        diffuse_iso = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model, name="isodiff",
                                apply_irf={"psf": False, "exposure": True, "edisp": True})
        
        diff_bkg = [diffuse_iem, diffuse_iso]
        return diff_bkg

    def _set_minmax(self, spectral_model, src):
        for par in src.params.keys():
            try:
                alt_par = paramCovf2g[par][0]
                factor = paramCovf2g[par][1]
            except:
                alt_par = par.lower()
                factor = 1

            if alt_par == "lambda_":
                temp = getattr(spectral_model, alt_par)
                temp.min = 20
                temp.max = 2000
            elif alt_par == "amplitude":
                temp = getattr(spectral_model, alt_par)
                temp.min = 1e-5*temp.scale
                temp.max = 1e5*temp.scale
            else:
                temp = getattr(spectral_model, alt_par)
                temp.min = float(src['spectral_pars'][par]['min'])*abs(src['spectral_pars'][par]['scale'])*factor
                temp.max = float(src['spectral_pars'][par]['max'])*abs(src['spectral_pars'][par]['scale'])*factor
                if (alt_par !='reference' and 
                    src.name != 'galdiff' and 
                    src.name != 'isodiff'):
                    temp.frozen = not(src['spectral_pars'][par]['free'])
        return spectral_model

    def initiateModel(self):
        self._models = Models(self._sources+self._diff_bkg)

    def _src_in_roi(self, ax):

        skydir = self.roi._src_skydir
        labels = [s.name for s in self.roi.point_sources]

        pixcrd = wcs_utils.skydir_to_pix(skydir, ax.wcs)
        path_effect = PathEffects.withStroke(linewidth=2.0, foreground="black")

        plot_kwargs = dict(linestyle='None', marker='+',
                                   markerfacecolor='None', mew=0.66, ms=10,
                                   markeredgecolor='w', clip_on=True)

        text_kwargs = dict(color='w', size=12, clip_on=True,
                                   fontweight='bold')

        for i, (x, y, label) in enumerate(zip(pixcrd[0], pixcrd[1], labels)):
            

            t = ax.annotate(label, xy=(x, y),
                            xytext=(5.0, 5.0), textcoords='offset points',
                            **text_kwargs)
            plt.setp(t, path_effects=[path_effect])

            t = ax.plot(x, y, **plot_kwargs)
            plt.setp(t, path_effects=[path_effect])


