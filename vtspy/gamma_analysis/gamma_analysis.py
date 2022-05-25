import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from astropy import units as u
from astropy.table import Table

from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from gammapy.modeling.models import PowerLawSpectralModel, Models
from gammapy.maps import Map, MapAxis
from gammapy.visualization import plot_spectrum_datasets_off_regions
from gammapy.estimators import FluxPointsEstimator
from gammapy.utils.scripts import make_path

from .fermi import FermiUtils
from .veritas import VeritasUtils

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

class GammaAnalysis(FermiUtils, VeritasUtils):
    def __init__(self, fermi_config='config_fermi.yaml', fermi_roi = "initial", fermi=False, veritas=False, verbosity=True, **kwargs):
        
        self._verbosity = verbosity
        self._overwrite = False

        if not(os.path.isdir("./results")):
            os.system("mkdir results")

        self._fermi = fermi
        self._veritas = veritas
        self._dataset = Datasets()
        
        fermi_range = [0.1, 300] * u.GeV
        veritas_range = [0.1, 100] * u.TeV

        emin = min(fermi_range[0] if fermi else 1*u.PeV, veritas_range[0] if veritas else 1*u.PeV)
        emax = max(fermi_range[-1]*fermi, veritas_range[-1]*veritas)
        self.energy_range = [emin, emax]

        if self.verbosity:
            print("[Log] Initializing gammapy...")

        if fermi:
            FermiUtils.__init__(self, file=fermi_config, roi = fermi_roi, verbosity=verbosity)
            self._default_fermi_setup()

        if veritas:
            VeritasUtils.__init__(self)
            
        if self.verbosity:
            print("[Log] Initialization of gammapy has been completed.")

    @property
    def verbosity(self):
        return self._verbosity
    
    @property
    def target(self):
        return self._target

    @property
    def target_id(self):
        return self._target_id
    
    @property
    def fit_result(self):
        return self._fit_result
    
    @property
    def datasets(self):
        return self._datasets

    @property
    def fermi(self):
        return self._fermi

    @property
    def veritas(self):
        return self._veritas

    @property
    def overwrite(self):
        return self._overwrite
    
    @property
    def models(self):
        return self._models

    @property
    def roi_model_name(self):
        names = [m.name for m in self._roi_models()]
        return names

    @property
    def flux_points(self):
        return self._flux_points

    @property
    def printModel(self):
        print(self.models)
    
    @property
    def printDatasets(self):
        print(self.datasets)

    @property
    def printParameter(self):
        for model in self.models:
            print(model.name, "*" if model.name == self.target.name else "", "(tentative TS = {:.2f}; offset = {:.3f} deg)".format(self.roi[model.name]['ts'], self.roi[model.name]['offset']))
            for par in model.parameters.names:
                print("\t {:10}".format(model.parameters[par].name), 
                      "*\t" if not(model.parameters[par].frozen) else "\t", 
                      "{:10.3f}".format(model.parameters[par].value) if model.parameters[par].name != "amplitude" else "{:10.1e}".format(model.parameters[par].value), 
                      "+/- {:10.3f}".format(model.parameters[par].error) if model.parameters[par].name != "amplitude" else "+/- {:10.1e}".format(model.parameters[par].error),
                      "\t",
                      "[{:10.3f}, {:10.3f}]".format(model.parameters[par].min, model.parameters[par].max) if model.parameters[par].name != "amplitude" else "[{:10.1e}, {:10.1e}]".format(model.parameters[par].min, model.parameters[par].max))
        
    @property
    def printAssociation(self):
        i = 1
        for src in self.roi.sources:
            
            if src.name == "isodiff" or src.name=="galdiff":
                continue
            
            if src.name==self.target.name:
                print(i, ":", src.name, " *")
            else:
                print(i, ":", src.name)
            print("\t", src.associations)
            i+=1

    @property
    def printTarget(self):
        print(self.models[self.target_id])
        print(self.models[self.target_id].spectral_model)


    def setTarget(self, target):
        if type(target)==int:
            self._target = self.roi.sources[target-1]
            self._target_id = target-1
            print("[Log] A target is set to", self.roi.sources[target-1].name)
        elif type(target)==str:
            i = 0
            for src in self.roi.sources:
                if src.name == "isodiff" or src.name=="galdiff":
                    continue
                elif target in src.associations:
                    self._target = self.roi.sources[i]
                    self._target_id = i
                    print("[Log] A target is set to", src.name)
                    break
                else:
                    i+=1
        
        self.freeSource(self.target.name)

    def setDatasets(self, fermi=False, veritas=False, mode='source'):
        self._datasets = Datasets()

        if fermi:
            self._datasets.append(self.FermiDatasets[0])
            if mode=="source":
                self._source_analysis_setup()

            self._datasets[-1].models = self.models

        if veritas:
            datasets_temp = Datasets()
            for dataset in self.VeritasDatasets:
                datasets_temp.append(dataset)
            
            self._datasets.append(datasets_temp.stack_reduce(name="VERITAS"))
            self._datasets[-1].models = self._roi_models()

    def setModels(self, telescope, models):
        new_datasets = Datasets()
        
        if (type(models) == str) and (models in self.models.names):
            models = self.models[models]
        elif type(models) != Models:
            print("[Error] The input model is not correct type (gammapy.modeling.models.Models or a model name).")
            return

        if telescope not in self.datasets.names:
            print("[Error] The input telescope is not in the list;", self.datasets.names)
            return

        for name in self.datasets.names:
            if name == telescope:
                if telescope == "Fermi-LAT":
                    new_datasets.append(self.FermiDatasets[0])
                    if mode=="source":
                        self._source_analysis_setup()
                    new_datasets[-1].models = models

                if telescope == "VERITAS":
                    datasets_temp = Datasets()
                    for dataset in self.VeritasDatasets:
                        datasets_temp.append(dataset)
                    
                    new_datasets.append(datasets_temp.stack_reduce(name="VERITAS"))
                    new_datasets[-1].models = models
            else:
                new_datasets.append(self.datasets[name])

        self._datasets = new_datasets
        

    def spectralAnalysis(self, joint=True, fermi=False, veritas=False):
        
        if fermi and not(self.fermi):
            print("[Error] Fermi-LAT is not initialized.")
            FermiUtils.__init__(self, file=fermi_config, roi = fermi_roi, verbosity=verbosity)
            self._default_fermi_setup()

        if veritas and not(self.veritas):
            print("[Error] VERITAS is not initialized.")            
            VeritasUtils.__init__(self)
        

        if not(fermi) and not(veritas):
            if len(self.datasets) == 0:
                print("[Error] None of datasets is selected (fermi=False and veritas=False).")  
                return
        elif len(self.datasets) == 0:
            self.setDatasets(fermi=fermi, veritas=veritas, joint=joint)

        self._Fit = Fit(self.datasets)
        self._fit_result = self._Fit.run()

        if self.verbosity:
            print(self.fit_result)

        if self._fit_result.success:
            self._models = self.datasets[0].models
            self._save_model()
            self._overwrite = True
    
    def calculateSED(self, name = None, load=True, **kwargs):

        if self.verbosity: 
            print("[Log] Initializing the flux calculation.")
        
        if not(getattr(self, "datasets")):
            print("[Error] Run 'setDatasets' first.")
            return
        
        overwrite = kwargs.get('overwrite', self.overwrite)

        if not(overwrite) and load and os.path.isfile("./results/flux_points.npy"):
            self._flux_points = np.load("./results/flux_points.npy", allow_pickle=True).item()
            self._load_model()
            if self.verbosity:
                print("[Log] The recent results (flux points and models) are loaded.")
        else:
            self._flux_points = {'fermi': {}, 'veritas': {}}
        
        if self.fermi:
            self._gammapy_2_fermipy_models(save=True)
            #self.freeSources(free=False)
            #self.freeSources(skydir=self.target.skydir, distance=[0.1], free=True, pars='norm')
            self.fit()
        if name == None:
            model_names = self.roi_model_name
        else:
            model_names = [ name ]

        for name in model_names:

            if self.fermi:
                if (name in self._flux_points['fermi'].keys()) and not(overwrite):
                    if self.verbosity: 
                        print("[Log][Fermi-LAT][{}] Already in 'flux_points'. ".format(name))
                        print("[Log] If you want to overwrite it, set overwrite=True.".format(name))
                    pass
                elif name in self.datasets["Fermi-LAT"].models.names:
                    if self.verbosity: 
                        print("[Log][Fermi-LAT][{}] Calculating...".format(name), end='\r')
                    self._flux_points['fermi'][name] = self.sed(name, loge_bins=[2.0,2.5,3.0,3.5,4.0,4.5,5.0], bin_index=2.2)
                    if self.verbosity: 
                        print("[Log][Fermi-LAT][{}] Completed.    ".format(name))

            if self.veritas:
                if (name in self._flux_points['veritas'].keys()) and not(overwrite):
                    if self.verbosity: 
                        print("[Log][VERITAS][{}] Already in 'flux_points'. ".format(name))
                        print("[Log] If you want to overwrite it, set overwrite=True.".format(name))
                    pass
                elif name in self.datasets["VERITAS"].models.names:
                    if self.verbosity: 
                        print("[Log][VERITAS][{}] Calculating...".format(name), end='\r')
                    energy_edges = MapAxis.from_bounds(0.1, 100, nbin=4, interp="log", unit="TeV").edges
                    self._flux_points['veritas'][name] = FluxPointsEstimator(
                                         energy_edges=energy_edges, source=name
                                         ).run([self.datasets["VERITAS"]]).table
                    if self.verbosity: 
                        print("[Log][VERITAS][{}] Completed.          ".format(name))
        self._overwrite=False
        
        np.save("./results/flux_points", self.flux_points)

    def plotRoI(self):
        plt.figure(figsize=(7, 7))
        
        if self.veritas:
            _, ax, _ = self._exclusion_mask.plot()
            self._on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="red")
            plot_spectrum_datasets_off_regions(ax=ax, datasets=self.VeritasDatasets)
        
        if self.fermi:
            if not(self.veritas):
                geom = Map.create(npix=(150, 150), binsz=0.05, skydir=self.target.skydir, proj="CAR", frame="icrs")
                _, ax, _ = geom.plot()
                ax.add_patch(Patches.Rectangle((0, 0), 150, 150,  color="w"))
            self._src_in_roi(ax)

        plt.show(block=False)

 

    def plotSED(self, name=None, ul=True):
        plt.figure(figsize=(8, 6))

        if not(getattr(self, "flux_points")):
            print("[Error] Run 'calculateSED' first.")
            return

        if name == None:
            ax = self._model_flx_plot()
            self._veritas_flux_plot(self._veritas_stack_flux(), ax = ax, label="VERITAS")
            self._fermi_flux_plot(self._fermi_stack_flux(), ax = ax, label="Fermi-LAT")
        else:
            ax = self._model_flx_plot(name)
            if self.veritas and (name in self._flux_points['veritas'].keys()):
                self._veritas_flux_plot(self._flux_points['veritas'][name], ul=ul, ax = ax, label="VERITAS")
            if self.fermi and  (name in self._flux_points['fermi'].keys()):
                self._fermi_flux_plot(self._flux_points['fermi'][name], ul=ul, ax = ax, TeV = self.veritas, label="Fermi-LAT")
        
        ax.legend(loc='upper right', frameon=False)
        ax.set_xlabel("Energy [TeV]", fontsize=12)
        ax.set_ylabel(r"E$^2$Flux [TeV/cm$^2$/s]", fontsize=12)
        ax.set_xlim(8e-5, 2e2)
        ax.set_ylim(1e-12, 1e-8)
        plt.show(block=False)
        

    def _model_flx_plot(self, name=None, **kwargs):
        
        ax = kwargs.pop('ax', plt.gca())
        
        if self.fermi and self.veritas:
            energy_range = [0.1, 100000] * u.GeV
        elif self.fermi and not(self.veritas):
            energy_range = [0.1, 300] * u.GeV
        else:
            energy_range = [0.1, 100] * u.TeV

        if name == None:
            energy_range = [0.1, 100000] * u.GeV
            e_edges = np.logspace(np.log10(energy_range[0]/u.GeV),np.log10(energy_range[1]/u.GeV), 100) 
            e_ctr = (e_edges[1:]+e_edges[:-1])/2. * 1e-3 * u.TeV
            total_model_flux = np.zeros(len(e_ctr))*u.TeV/u.cm**2/u.s
        
            for name in self.roi_model_name:
                src_spec = self.datasets[0].models[name].spectral_model
                total_model_flux+=src_spec(e_ctr).to("cm-2 s-1 TeV-1")*np.power(e_ctr,2)
                ax = src_spec.plot(energy_range=energy_range, energy_power=2, label=name)
                src_spec.plot_error(ax=ax, energy_range=energy_range, energy_power=2)
            ax.plot(e_ctr, total_model_flux, color="k")
        else:
            src_spec = self.datasets[0].models[name].spectral_model
            ax = src_spec.plot(energy_range=energy_range, energy_power=2, label=name)
            src_spec.plot_error(ax=ax, energy_range=energy_range, energy_power=2)
        
        return ax

    def _veritas_flux_plot(self, sed, ul=True, **kwargs):
        ax = kwargs.pop('ax', plt.gca())

        kw = {}
        kw['marker'] = kwargs.get('marker', 'x')
        kw['linestyle'] = kwargs.get('linestyle', 'None')
        kw['label'] = kwargs.get('label', None)
        
        m = ~((sed['ts'] > 4) * (sed['success']))
        x = sed['e_ref']
        y = sed['dnde']*sed['e_ref']**2

        yerr = sed['dnde_err']*sed['e_ref']**2
        yul = sed['dnde_ul']*sed['e_ref']**2

        delo = sed['e_ref'] - sed['e_min']
        dehi = sed['e_max'] - sed['e_ref']
        xerr0 = np.vstack((delo[m], dehi[m]))
        xerr1 = np.vstack((delo[~m], dehi[~m]))

        etc = plt.errorbar(x[~m], y[~m], xerr=xerr1,
                     yerr=yerr[~m], **kw)
        if ul:
            kw['label'] = None
            plt.errorbar(x[m], yul[m], xerr=xerr0,
                     yerr=yul[m] * 0.2, uplims=True, color=etc.lines[0].get_c(), **kw)

        ax.set_yscale('log')
        ax.set_xscale('log')

        return ax

    def _fermi_flux_plot(self, sed, TeV=True, ul=True, **kwargs):
        ax = kwargs.pop('ax', plt.gca())

        if TeV:
            factor=1e-6
        else:
            factor=1

        kw = {}
        kw['marker'] = kwargs.get('marker', 'x')
        kw['linestyle'] = kwargs.get('linestyle', 'None')
        kw['label'] = kwargs.get('label', None)
        
        m = sed['ts'] < 4
        x = sed['e_ctr']
        y = sed['e2dnde']
        yerr = sed['e2dnde_err']
        yerr_lo = sed['e2dnde_err_lo']
        yerr_hi = sed['e2dnde_err_hi']
        yul = sed['e2dnde_ul95']

        delo = sed['e_ctr'] - sed['e_min']
        dehi = sed['e_max'] - sed['e_ctr']
        xerr0 = np.vstack((delo[m], dehi[m]))
        xerr1 = np.vstack((delo[~m], dehi[~m]))

        etc = plt.errorbar(x[~m]*factor, y[~m]*factor, xerr=xerr1*factor,
                     yerr=(yerr_lo[~m]*factor, yerr_hi[~m]*factor), ls="", **kw)
        if ul:
            kw['label'] = None
            plt.errorbar(x[m]*factor, yul[m]*factor, xerr=xerr0*factor,
                     yerr=yul[m] * 0.2 * factor, uplims=True, ls="", color=etc.lines[0].get_c(), **kw)

        ax.set_yscale('log')
        ax.set_xscale('log')
        return ax

    def _fermi_stack_flux(self):
        flux_stack = Table()
        model_names = list(self.flux_points['fermi'].keys())
        flux_points = self.flux_points['fermi']
        data_size = len(flux_points[model_names[0]]['e_ctr'])
        flux_stack['ts'] = np.zeros(data_size)
        flux_stack['e_ctr'] = flux_points[model_names[0]]['e_ctr']
        flux_stack['e_min'] = flux_points[model_names[0]]['e_min']
        flux_stack['e_max'] = flux_points[model_names[0]]['e_max']
        flux_stack['e2dnde'] = np.zeros(data_size)
        flux_stack['e2dnde_err'] = np.zeros(data_size)
        flux_stack['e2dnde_err_lo'] = np.zeros(data_size)
        flux_stack['e2dnde_err_hi'] = np.zeros(data_size)
        flux_stack['e2dnde_ul95'] = np.zeros(data_size)

        for name in model_names:
            flux_stack['ts'] = [max(x, y) for x, y in zip(flux_stack['ts'], flux_points[name]['ts'])]
            flux_stack['e2dnde'] += flux_points[name]['e2dnde']
            flux_stack['e2dnde_err'] = np.sqrt(flux_stack['e2dnde_err']**2.+flux_points[name]['e2dnde']**2.)
            flux_stack['e2dnde_err_lo'] = np.sqrt(np.nan_to_num(flux_stack['e2dnde_err_lo'])**2.+np.nan_to_num(flux_points[name]['e2dnde_err_lo'])**2.)
            flux_stack['e2dnde_err_hi'] = np.sqrt(np.nan_to_num(flux_stack['e2dnde_err_hi'])**2.+np.nan_to_num(flux_points[name]['e2dnde_err_hi'])**2.)
            flux_stack['e2dnde_ul95'] += flux_points[name]['e2dnde_ul95']
        return flux_stack

    def _veritas_stack_flux(self):
        flux_stack = Table()
        model_names = list(self.flux_points['veritas'].keys())
        flux_points = self.flux_points['veritas']
        data_size = len(flux_points[model_names[0]]['e_ref'])
        flux_stack['ts'] = np.zeros(data_size)
        flux_stack['success'] = np.zeros(data_size)
        flux_stack['e_ref'] = flux_points[model_names[0]]['e_ref']
        flux_stack['e_min'] = flux_points[model_names[0]]['e_min']
        flux_stack['e_max'] = flux_points[model_names[0]]['e_max']
        flux_stack['dnde'] = np.zeros(data_size)
        flux_stack['dnde_err'] = np.zeros(data_size)
        flux_stack['dnde_ul'] = np.zeros(data_size)
        
        for name in model_names:
            flux_stack['ts'] = [max(x, y) for x, y in zip(flux_stack['ts'], flux_points[name]['ts'])]
            flux_stack['success'] = [bool(x+y) for x, y in zip(flux_stack['success'], flux_points[name]['success'])]
            flux_stack['dnde'] += flux_points[name]['dnde']
            flux_stack['dnde_err'] = np.sqrt(np.nan_to_num(flux_stack['dnde_err'])**2.+np.nan_to_num(flux_points[name]['dnde_err'])**2.)
            flux_stack['dnde_ul'] += flux_points[name]['dnde_ul']
        return flux_stack

    def _save_model(self, path = "./results/", file = "models.yaml"):
        path = make_path(path)
        self.models.write(path/file, full_output=True, overwrite=True)
        diff_bkg = []
        if 'galdiff' in self.models.names:
            diff_bkg.append(self.models['galdiff'])
        if 'isodiff' in self.models.names:
            diff_bkg.append(self.models['isodiff'])
        self._diff_bkg = diff_bkg

    def _load_model(self, path = "./results/", file = "models.yaml"):
        path = make_path(path)
        data = yaml.safe_load(open(path/file))
        for src in data['components']:
            name = src['name']
            model = self.models[name]
            if name in self.models.names:
                model.spectral_model = model.spectral_model.from_dict(src['spectral'])
            else:
                print("[Error] {} is not exist in the previous result. Run 'spectralAnalysis' again".format(src['name']))        
        covfile = file.replace(".yaml", "_covariance.dat")
        self.models.read_covariance(path, filename=covfile, format="ascii.fixed_width")