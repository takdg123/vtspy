import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
import os
import pickle
import copy
import glob

from . import FermiAnalysis, VeritasAnalysis
from ..utils import logger
from .. import utils
from ..model import default_model
from .. import plotting
from ..config import JointConfig


from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.modeling import Fit
import gammapy.modeling.models as gammapy_model
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.modeling.models import SkyModel


class JointAnalysis:
    """
    This is to perform a joint VERITAS and Fermi-LAT analysis. This class
    exploits results from VeritasAnalysis and FermiAnalysis.

    Args:
        veritas (str or vtspy.VeritasAnalysis): state filename or class for VERITAS
            Default: initial
        fermi (str or vtspy.FermiAnalysis): state filename or class for Fermi-LAT
            Default: initial
        config_file (str): config filename (yaml)
            Default: config.yaml
        verbosity (int)
    """


    def __init__(self, veritas = None, fermi = None, config_file='config.yaml', verbosity=1, **kwargs):
        self._verbosity = verbosity
        self._logging = logger(self.verbosity)
        self._logging.info("Initialize the joint-fit analysis...")
        self.config = JointConfig.get_config(config_file).pop("joint")
        self._outdir = kwargs.pop("outdir", self.config["fileio"]["outdir"])

        if not(os.path.isdir(self._outdir)):
            os.system(f"mkdir {self._outdir}")

        self._model_change_flag = False
        self._fit_flag = False

        if type(veritas) == str:
            self.veritas = VeritasAnalysis(veritas, config_file=config_file)
            self._veritas_state = veritas
            self._target_name = self.veritas.target_name
        elif hasattr(veritas, "datasets"):
            self._logging.info("VERITAS datasets is imported.")
            self.veritas = veritas
            self._veritas_state = self.veritas._veritas_state
            self._target_name = self.veritas.target_name

        if type(fermi) == str:
            self.fermi = FermiAnalysis(fermi, construct_dataset=True, config_file=config_file)
            self._fermi_state = fermi
        elif hasattr(fermi, "gta"):
            self.fermi = fermi
            self._fermi_state = self.fermi._fermi_state
            self.fermi.construct_dataset()
            self._logging.info("Fermi-LAT datasets is imported.")

        if hasattr(self, "fermi") and hasattr(self, "veritas"):
            self._logging.info("Constructing a joint datasets")
            self._construct_joint_datasets()
        self._logging.info("Completed.")

    @property
    def target_model(self):
        """
        Return:
            gammapy.modeling.models.SkyModel
        """
        return self.datasets.models[self.target_name]

    @property
    def verbosity(self):
        """
        Return:
            int
        """
        return self._verbosity

    @property
    def target_name(self):
        """
        Return:
            str: target name
        """
        return self._target_name

    def print_datasets(self):
        """
        Print datasets

        Return:
            astropy.table
        """

        return self._logging.info(self.datasets)


    def print_models(self, full_output=False):
        """
        Print model and parameters

        Args:
            full_output (bool): return a target model or all models
                default: False
        Return:
            astropy.table
        """
        if full_output:
            return self._logging.info(self.datasets.models)
        else:
            table = self.datasets.models.to_parameters_table()
            table = table[table["model"]== self.target_name]
            return table

    def save_state(self, state_file):
        """
        Save the state

        Args:
            state_file (str): the name of state
        """

        filename = f"./{self._outdir}/{state_file}.pickle".format(state_file)
        with open(filename, 'wb') as file:
            temp = [copy.copy(self.veritas), copy.copy(self.fermi)]
            del(self.veritas)
            del(self.fermi)
            del(self._logging)
            pickle.dump(self, file)
            self._logging = logger(self.verbosity)
            self.veritas, self.fermi = temp

    def load_state(self, state_file):
        """
        Load the state

        Args:
        state_file (str): the name of state
        """
        try:
            filename = f"./{self._outdir}/{state_file}.pickle".format(state_file)
            with open(filename, 'rb') as file:
                self.__dict__.update(pickle.load(file).__dict__)

            if not(hasattr(self, "fermi")):
                self.fermi = FermiAnalysis(self._fermi_state, construct_dataset=True)

            if not(hasattr(self, "veritas")):
                self.veritas = VeritasAnalysis(self._veritas_state)

        except:
            self._logging.error("The state file does not exist. Check the name again")
            return -1

    def list_of_state(self):
        files = glob.glob(f"{self._outdir}/*.pickle")
        return print([n.split("/")[-1].split(".")[0] for n in files])

    def optimize(self, method="flux", model=None, instrument="VERITAS", **kwargs):
        """
        To find good initial parameters for a given model.

        Args:
            method (str): either "flux", "rough" or "inst". The "flux" method will fit flux points
                from VeritasAnalysis.analysis("sed") and FermiAnalysis.analysis("sed"). The "rough"
                method will fit the dataset with tol of 1 and strategy of 1 (fast). The
                "inst" method will fit the model with one of datasets (defined by instrument)
                Default: flux
            model (gammapy.modeling.models.SpectralModel): a model to fit
                Default: None (fit with the current target model)
            instrument(str): instrument used for the "inst" method
                Default: VERITAS
            **kwargs
        """

        if model is None:
            model = self.datasets.models[self.target_name].spectral_model

        if method == "rough":
            
            optimize_opts_default = {
                "method": "L-BFGS-B",
                "options": {"ftol": 1e-4, "gtol": 1e-05, "maxls": 40},
                "backend": "scipy",
            }
            optimize_opts = kwargs.pop("optimize_opts", optimize_opts_default)

            fit_ = Fit(optimize_opts = optimize_opts)
            fit_ = fit_.run(self.datasets)

        elif method == "flux":
            test_model = SkyModel(spectral_model=model, name="test")
            fermi_sed = kwargs.pop("fermi_sed", f"{self.fermi._outdir}/{self._fermi_state}_sed.fits")

            if not(os.path.isfile(fermi_sed)):
                self.fermi.analysis("sed", state_file = self._fermi_state)

            table = Table.read(fermi_sed, format='fits', hdu=1)
            data = FluxPoints.from_table(table, sed_type="likelihood")

            fermi_dataset = FluxPointsDataset(data=data, models=test_model)
            fermi_dataset.mask_safe = ~fermi_dataset.data.to_table()["is_ul"]
            fermi_dataset.mask_fit = ~fermi_dataset.data.to_table()["is_ul"]

            veritas_dataset = FluxPointsDataset(data=self.veritas.flux_points, models=test_model)
            nan_norm = ~np.isnan(veritas_dataset.data.to_table()["norm"])
            veritas_dataset.mask_safe = veritas_dataset.mask_safe*nan_norm
            veritas_dataset.mask_fit = veritas_dataset.mask_safe*nan_norm

            self._optimize_flux_datasets = Datasets([fermi_dataset, veritas_dataset])
            
            self._optimize_flux_datasets.models = test_model
            
            optimize_opts_default = {
                "method": "L-BFGS-B",
                "backend": "scipy",
            }

            optimize_opts = optimize_opts_default

            fit_ = Fit(optimize_opts =  optimize_opts)
            prev_stats = 1e10
            num_run = 0
            while True:
                result_optimize = fit_.run(self._optimize_flux_datasets)
                current_stats = result_optimize.total_stat
                if abs((current_stats-prev_stats)/current_stats) < 0.1 :
                    break
                else:
                    prev_stats = current_stats
                    num_run+=1
            self.datasets.models[self.target_name].spectral_model = test_model.spectral_model
            self._logging.info(f"Optimize {num_run} times to get an inital parameters.")

        elif method == "inst":
            self.datasets.models[self.target_name].spectral_model = model
            joint_fit = Fit()
            fit_results = joint_fit.run(self.datasets[instrument.lower()])
        else:
            return

    def fit(self, **kwargs):
        """
        Perform a joint-fit analysis

        Args:
            **kwargs: passed to vtspy.JointAnalysis.optimize
        """

        optimize = kwargs.pop("optimize", True)
        if self._model_change_flag and optimize:
            self._logging.info("A model is recently updated. Optimizing the input parameters...")
            self.optimize(**kwargs)
            self._model_change_flag = False
            self._logging.info("Completed. Move to the next step.")

        self._logging.info("Start fitting...")
        
        joint_fit = Fit()
        self.fit_results = joint_fit.run(self.datasets)

        if self.fit_results.success:
            self._logging.info("Fit successfully.")
            self._fit_flag = True
        else:
            self._logging.error("Fit fails.")

    def analysis(self, **kwargs):

        """
        Perform a SED analysis.

        Args:
            **kwargs: passed to FluxPointsEstimator
        """

        energy_bins = kwargs.get("energy_bins", np.geomspace(0.0001, 10, 20) * u.TeV)

        fpe = FluxPointsEstimator(
            energy_edges=energy_bins,
            source=self.target_name, selection_optional="all"
            )

        self.flux_points = fpe.run(self.datasets)

        self._flux_points_dataset = FluxPointsDataset(
            data=self.flux_points, models=self.datasets.models
        )


    def plot(self, output, **kwargs):
        """
        Show various results: SED

        Args:
            output (str): a plot to show
                Options: ["sed"]
            **kwargs: passed to vtspy.JointAnalysis.plot_sed
        """

        if output == "sed":
            self.plot_sed(**kwargs)


    def plot_sed(self, fermi=True, veritas=True, joint=True, show_model = True, show_flux_points=True, **kwargs):
        """
        Plot a spectral energy distribution (SED) with a model and flux points.

        Args:
            fermi (bool): show Fermi-LAT results
                Default: True
            veritas (bool): show VERITAS results
                Default: True
            fermi (bool): show Joint-fit results
                Default: True
            show_flux_points (bool): slow flux points
                Default: True
            show_model (bool) show model
                Default: True
            **kwargs: passed to plotting.plot_sed
        """

        if fermi and not(hasattr(self.fermi, "output")):
            self.fermi.simple_analysis("sed")

        if veritas and not(hasattr(self.veritas, "_flux_points_dataset")):
            self.veritas.simple_analysis()

        if joint and not(self._fit_flag):
            fit = False
        else:
            fit = True

        cmap = plt.get_cmap("tab10")
        i = 0

        if veritas:
            vts = self.veritas._flux_points_dataset

            if show_flux_points:
                vts.data.plot(sed_type="e2dnde", color = cmap(i), label="VERITAS")

            if not(fit) and show_model:
                veritas_model = vts.models[0].spectral_model
                veritas_model.plot(energy_bounds=vts._energy_bounds, sed_type="e2dnde", color=cmap(i))
                veritas_model.plot_error(energy_bounds=vts._energy_bounds,
                                         sed_type="e2dnde", alpha=0.2, facecolor=cmap(i))
            i+=1

        if fermi:
            plotting.plot_sed(self.fermi.output, units="TeV", erg=True, show_flux_points=show_flux_points,
                show_model = not(fit)*show_model, color=cmap(i), **kwargs)
            i+=1

        emin = 1*u.TeV
        emax = 1*u.TeV
        for dataset in self.datasets:
            if (dataset.name != "fermi") and (dataset.name != "veritas"):
                self.plot_flux(dataset, color=cmap(i))
                i+=1

            if type(dataset) == FluxPointsDataset:
                emin_d = dataset.data.energy_min[0]
                emax_d = dataset.data.energy_max[-1]
            else:
                emin_d = dataset.energy_range_total[0]
                emax_d = dataset.energy_range_total[1]
            
            emin = min(emin, emin_d/2.)
            emax = max(emax, emax_d*2.)
            
        energy_bounds = [emin, emax]
        if joint and show_model:
            jf_model = self.datasets.models[self.target_name].spectral_model
                
            if fit:
                jf_model.plot(energy_bounds=energy_bounds, sed_type="e2dnde", color=cmap(i), label=self.target_name)
                jf_model.plot_error(energy_bounds=energy_bounds,
                                         sed_type="e2dnde", alpha=0.2, color="k")
            else:
                jf_model.plot(energy_bounds=energy_bounds, sed_type="e2dnde", color=cmap(i), label="Before fit", ls="--")
            i+=1

        plt.xlim(emin, emax)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=13)
        plt.grid(which="major", ls="-")
        plt.grid(which="minor", ls=":")
        plt.xlabel("Energy [TeV]", fontsize=13)
        plt.ylabel(r"Energy flux [erg/cm$^2$/s]", fontsize=13)


    def change_model(self, model, optimize=False, **kwargs):
        """
        To change a target model

        Args:
            model (str or gammapy.modeling.models.SpectralModel): a new target model
            optimize (bool): perform optimization (JointAnalysis.optimize)
                Default: False
            **kwargs: passed to JointAnalysis.optimize
        """

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
            self.optimize(model=spectral_model, **kwargs)
            self._model_change_flag = False
        else:
            self._model_change_flag = True
            self.datasets.models[self.target_name].spectral_model = spectral_model

        newmodel = self.datasets.models[self.target_name].spectral_model.tag[0]
        self._fit_flag = False
        self._logging.info(f"The spectral model for the target is changed:")
        self._logging.info(f"{prevmodel}->{newmodel}")

    def add_dataset(self, data, sync = True, model=None, **kwargs):
        """
        To add a new dataset.

        Args:
            data (str or gammapy.estimators.FluxPoints): new dataset
            sync (bool): synchronize with the target model
                Default: True
            model (gammapy.modeling.models.SpectralModel): a model for the new dataset
            **kwargs: passed to gammapy.estimators.FluxPointsDataset
        """
        if type(data) == str:
            table = Table.read(data)
            data = FluxPoints.from_table(table, sed_type="e2dnde")
        
        if type(data) == FluxPoints:
            if sync:
                model = self.target_model
            else:
                if model is None:
                    model = gammapy_model.PowerLawSpectralModel()
                    self._logging.warning(f"The model is assumed to be a power law.")
                target_name = kwargs.pop("target_name", "new component")
                model = SkyModel(spectral_model=model, name=target_name)
            name = kwargs.pop("name", f"dataset_{len(self.datasets)}")
            new_dataset = FluxPointsDataset(data=data, models=model, name=name, **kwargs)
            self.datasets.append(new_dataset)
        else:
            self._logging.error(f"This data type is not supported yet.")
            
    def _construct_joint_datasets(self, inst="VERITAS"):
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

    def plot_flux(self, dataset, **kwargs):
        if type(dataset) == FluxPointsDataset:
            dataset.data.plot(label=dataset.name, energy_power=0, **kwargs)
