import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from . import FermiAnalysis, VeritasAnalysis
from ..utils import logger
from .. import utils
from ..model import default_model

from gammapy.datasets import Datasets
from gammapy.modeling import Fit
import gammapy.modeling.models as gammapy_model
from gammapy.estimators import FluxPointsEstimator


class JointAnalysis:
    
    def __init__(self, veritas = "initial", fermi = "initlal", verbosity=1):
        self._verbosity = verbosity
        self._logging = logger(self.verbosity)
        self._logging.info("Initialize the joint-fit analysis...")
        self._outdir = "./joint/"
        self._model_change_flag = False
        

        if type(veritas) == str:
            self.veritas = VeritasAnalysis(veritas)
        elif hasattr(veritas, "datasets"):
            self._logging.info("VERITAS datasets is imported.")
            self.veritas = veritas
        else:
            return
        self._target_name = self.veritas.target_name

        if type(fermi) == str:
            self.fermi = FermiAnalysis(fermi, construct_dataset=True)
        elif hasattr(fermi, "datasets"):
            self._logging.info("Fermi-LAT datasets is imported.")
            self.fermi = fermi
        else:
            return
        
        self._logging.info("Constructing a joint datasets")
        self._construct_joint_datasets(init=True)
        self._logging.info("Completed.")
        
    @property
    def target_model(self):
        return self.datasets.models[self.target_name]
    
    @property
    def verbosity(self):
        return self._verbosity

    @property
    def target_name(self):
        return self._target_name
    
    def print_datasets(self):
        return self._logging.info(self.datasets)

    
    def print_models(self, full_output=False):
        if full_output:
            return self._logging.info(self.datasets.models)
        else:
            table = self.datasets.models.to_parameters_table()
            table = table[table["model"]== self.target_name]
            return table
    
    def fit(self, **kwargs):

        optimize = kwargs.pop("optimize", True)
        if self._model_change_flag and optimize:
            self._logging.info("A model is recently updated. Optimizing the input parameters...")
            self._optimize()
            self._logging.info("Completed. Move to the next step.")

        self._logging.info("Start fitting...")

        joint_fit = Fit()
        self.fit_results = joint_fit.run(self.datasets)

        if self.fit_results.success:
            self._logging.info("Fit successfully.")
        else:
            self._logging.error("Fit fails.")
        
    def sed_plot(self, fermi=True, veritas=True, joint=True, **kwargs):

        show_flux_points=True
        
        if fermi and not(hasattr(self.fermi, "output")):
            self.fermi.simple_analysis("sed")
        
        if veritas and not(hasattr(self.veritas, "_flux_points_dataset")):
            self.veritas.simple_analysis()
        
        if joint and not(hasattr(self, "fit_results")):
            fit = False
        else:
            fit = True
            show_flux_points = kwargs.pop("show_flux_points", False)

        cmap = plt.get_cmap("tab10")
        i = 0

        if joint:
            if fit:
                energy_bounds = [100 * u.MeV, 30 * u.TeV]
                jf_model = self.datasets.models[0].spectral_model
                jf_model.plot(energy_bounds=energy_bounds, sed_type="e2dnde", color=cmap(i), label=self.target_name)
                jf_model.plot_error(energy_bounds=energy_bounds, 
                                         sed_type="e2dnde", alpha=0.2, color="k")
            else:
                energy_bounds = [100 * u.MeV, 30 * u.TeV]
                jf_model = self.datasets.models[self.target_name].spectral_model
                
                if fit:
                    jf_model.plot(energy_bounds=energy_bounds, sed_type="e2dnde", color=cmap(i), label=self.target_name, ls="-")
                    jf_model.plot_error(energy_bounds=energy_bounds, 
                                         sed_type="e2dnde", alpha=0.2, color="k")
                else:
                    jf_model.plot(energy_bounds=energy_bounds, sed_type="e2dnde", color=cmap(i), label="Before fit", ls="--")
            i+=1

        if veritas:
            
            vts = self.veritas._flux_points_dataset
            energy_bounds = vts._energy_bounds
            if show_flux_points:
                vts.data.plot(sed_type="e2dnde", color = cmap(i), label="VERITAS")

            if not(fit):
                veritas_model = vts.models[0].spectral_model
                veritas_model.plot(energy_bounds=energy_bounds, sed_type="e2dnde", color=cmap(i))
                veritas_model.plot_error(energy_bounds=energy_bounds, 
                                         sed_type="e2dnde", alpha=0.2, color="k")
            i+=1

        if fermi:
            fermi_model = self.fermi.output["sed"]['model_flux']

            m_engs = 10**fermi_model['log_energies']
            to_TeV = 1e-6

            e2 = m_engs**2.*utils.MeV2Erg

            sed = self.fermi.output["sed"]
            ul_ts_threshold = kwargs.pop('ul_ts_threshold', 4)
            m = sed['ts'] < ul_ts_threshold
            x = sed['e_ctr']*to_TeV
            y = sed['e2dnde']*utils.MeV2Erg

            yerr = sed['e2dnde_err']*utils.MeV2Erg
            yerr_lo = sed['e2dnde_err_lo']*utils.MeV2Erg
            yerr_hi = sed['e2dnde_err_hi']*utils.MeV2Erg
            yul = sed['e2dnde_ul95']*utils.MeV2Erg
            delo = sed['e_ctr'] - sed['e_min']
            dehi = sed['e_max'] - sed['e_ctr']
            xerr0 = np.vstack((delo[m], dehi[m]))*to_TeV
            xerr1 = np.vstack((delo[~m], dehi[~m]))*to_TeV
            if show_flux_points:
                plt.errorbar(x[~m], y[~m], xerr=xerr1, label="Fermi-LAT",
                             yerr=(yerr_lo[~m], yerr_hi[~m]), ls="", color=cmap(i))
                plt.errorbar(x[m], yul[m], xerr=xerr0,
                             yerr=yul[m] * 0.2, uplims=True, ls="", color=cmap(i))

            if not(fit):
                plt.plot(m_engs*to_TeV, fermi_model['dnde'] * e2, color=cmap(i))
                plt.fill_between(m_engs*to_TeV, fermi_model['dnde_lo'] * e2, fermi_model['dnde_hi'] * e2,
                alpha=0.2, color="k")

            plt.xlim(5e-5, 30)
            i+=1

        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=13)
        plt.grid(which="major", ls="-")
        plt.grid(which="minor", ls=":")
        plt.xlabel("Energy [TeV]", fontsize=13)
        plt.ylabel(r"Energy flux [erg/cm$^2$/s]", fontsize=13)

    def analysis(self, **kwargs):
        
        energy_bins = kwargs.get("energy_bins", np.geomspace(0.0001, 10, 20) * u.TeV)

        fpe = FluxPointsEstimator(
            energy_edges=energy_bins, 
            source=self.target_name, selection_optional="all", **kwargs
            )

        self.flux_points = fpe.run(self.datasets)

        self._flux_points_dataset = FluxPointsDataset(
            data=self.flux_points, models=self.datasets.models
        )
        
    def change_model(self, model, optimize=False, **kwargs):
        prevmodel = self.datasets.models[self.target_name].spectral_model.tag[0]
        if type(model) == str:
            spectral_model = default_model(model, **kwargs)
            if model is None:
                self._logging.error("The input model is not supported yet.")
                return
        elif hasattr(model, "tag"):
            spectral_model = model
            self._logging.info(f"A model, {model.tag[0]}, is imported")
        
        if optimize:
            self._optimize(model=spectral_model)
            self._model_change_flag = False
        else:
            self._model_change_flag = True
            self.datasets.models[self.target_name].spectral_model = spectral_model

        newmodel = self.datasets.models[self.target_name].spectral_model.tag[0]
        self._logging.info(f"The spectral model for the target is chaged:")
        self._logging.info(f"{prevmodel}->{newmodel}")

    def _optimize(self, model=None):
        if model is None:
            model = self.datasets.models[self.target_name].spectral_model
        
        self.datasets.models[self.target_name].spectral_model = model
        joint_fit = Fit()
        fit_results = joint_fit.run(self.datasets["veritas"])

    def _construct_joint_datasets(self, inst="VERITAS", init=False):
        vts_model = self.veritas.stacked_dataset.models[0]
        fermi_model = self.fermi.datasets.models[self.fermi.target_name]

        self.veritas.stacked_dataset.models = self._find_target_model()
        self.datasets = Datasets([self.fermi.datasets, self.veritas.stacked_dataset])

        if inst.lower() == "veritas":
            self.datasets.models[self.fermi.target_name].spectral_model = vts_model.spectral_model
        elif inst.lower() == "fermi":
            self.datasets.models[self.veritas.target_name].spectral_model = fermi_model.spectral_model

        self.datasets.models[self.fermi.target_name]._name = self.target_name

    def _find_target_model(self):
        target_pos = self.fermi.datasets.models[self.fermi.target_name].spatial_model.position
        th2cut = self.veritas._on_region.radius.value

        models = []
        for model in self.fermi.datasets.models:
            if model.name != 'galdiff' and model.name != 'isodiff':
                if target_pos.separation(model.spatial_model.position).deg < th2cut:
                    models.append(model)
        return models
    
