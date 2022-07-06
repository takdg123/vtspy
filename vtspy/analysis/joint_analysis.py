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
from gammapy.modeling.models import SkyModel, DatasetModels



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
        **kwargs: passed to JointAnalysis.construct_dataset
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
        self._num_of_models = 1
        self._target_name = ['','']
        if type(veritas) == str:
            self.veritas = VeritasAnalysis(veritas, config_file=config_file)
            self._veritas_state = veritas
            self._target_name[0] = self.veritas.target_name
        elif hasattr(veritas, "datasets"):
            self._logging.info("VERITAS datasets is imported.")
            self.veritas = veritas
            self._veritas_state = self.veritas._veritas_state
            self._target_name[0] = self.veritas.target_name

        if type(fermi) == str:
            self.fermi = FermiAnalysis(fermi, construct_dataset=True, config_file=config_file)
            self._fermi_state = fermi
            self._target_name[1] = self.fermi.target_name
        elif hasattr(fermi, "gta"):
            self.fermi = fermi
            self._fermi_state = self.fermi._fermi_state
            self.fermi.construct_dataset()
            self._logging.info("Fermi-LAT datasets is imported.")
            self._target_name[1] = self.fermi.target_name

        if hasattr(self, "fermi") and hasattr(self, "veritas"):
            self._logging.info("Constructing a joint datasets")
            self.construct_dataset(**kwargs)

        self._logging.info("Completed.")

    @property
    def target_model(self):
        """
        Return:
            gammapy.modeling.models.SkyModel
        """
        return self.datasets["veritas"].models

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
            tuple: target name (veritas and fermi)
        """
        return tuple(self._target_name)


    @property
    def _target_spatial_model(self):
        return gammapy_model.PointSpatialModel(
        lon_0="{:.5f} deg".format(self.fermi.target.radec[0]), lat_0="{:.3f} deg".format(self.fermi.target.radec[1]), frame="icrs"
        )

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
            table = self.datasets["veritas"].models.to_parameters_table()
            return table[table["frozen"]==False]

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

    def load_state(self, state_file, reconstruct=False):
        """
        Load the state

        Args:
        state_file (str): the name of state
        reconstruct (bool): re-construct the datasets
            Default: False
        """
        filename = f"./{self._outdir}/{state_file}.pickle".format(state_file)
        
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.__dict__.update(pickle.load(file).__dict__)
            self._target_name = ['', '']
            
            if not(hasattr(self, "fermi")):
                self.fermi = FermiAnalysis(self._fermi_state, construct_dataset=True)
                self._target_name[1] = self.fermi.target_name

            if not(hasattr(self, "veritas")):
                self.veritas = VeritasAnalysis(self._veritas_state)
                self._target_name[0] = self.veritas.target_name

            if reconstruct:
                self.construct_dataset()
        else:
            self._logging.error("The state file does not exist. Check the name again")
            return -1

    def list_of_state(self):
        files = glob.glob(f"{self._outdir}/*.pickle")
        return print([n.split("/")[-1].split(".")[0] for n in files])

    def optimize(self, method="flux", instrument="VERITAS", **kwargs):
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

        if method == "rough":
            optimize_opts = kwargs.pop("optimize_opts", optimize_opts_default)

            fit_ = Fit(optimize_opts = optimize_opts)
            fit_ = fit_.run(self.datasets)
            self._model_change_flag=False

        elif method == "flux":
            
            test_model = copy.copy(self.target_model)
            fermi_dataset = self._fermi_flux_dataset(models=test_model)

            veritas_dataset = FluxPointsDataset(data=self.veritas.flux_points, models=test_model)
            nan_norm = ~np.isnan(veritas_dataset.data.to_table()["norm"])
            veritas_dataset.mask_safe = veritas_dataset.mask_safe*nan_norm
            veritas_dataset.mask_fit = veritas_dataset.mask_safe*nan_norm

            self._optimize_flux_datasets = Datasets([fermi_dataset, veritas_dataset])
            
            self._optimize_flux_datasets.models = test_model

            for dataset in self.datasets:
                if (dataset.name != "fermi") and (dataset.name != "veritas"):
                    if type(dataset) == FluxPointsDataset:
                        self._optimize_flux_datasets.append(dataset)

            optimize_opts_default = {
                "method": "L-BFGS-B",
                "backend": "scipy",
            }

            optimize_opts = optimize_opts_default

            fit_ = Fit(optimize_opts =  optimize_opts)
            prev_stats = -1
            num_run = 0
            while True:
                _optimize_result = fit_.run(self._optimize_flux_datasets)
                current_stats = _optimize_result.total_stat
                if prev_stats == -1:
                    self._logging.info(f"Initial -> {current_stats}")
                    prev_stats = current_stats
                    num_run+=1
                elif (abs((current_stats-prev_stats)/current_stats) < 0.1) and (_optimize_result.success):
                    break
                else:
                    self._logging.info(f"{prev_stats} -> {current_stats}")
                    prev_stats = current_stats
                    num_run+=1

            self._logging.debug(_optimize_result)
            self._optimize_result = _optimize_result

            for model in test_model:
                self.datasets.models[model.name].spectral_model = model.spectral_model

            self._logging.info(f"Optimize {num_run} times to get an inital parameters.")
            self._model_change_flag=False
        elif method == "inst":
            joint_fit = Fit()
            fit_results = joint_fit.run(self.datasets[instrument.lower()])
            self._model_change_flag=False
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
        
        joint_fit = Fit(store_trace=True)
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
            source=self.target_name[1], selection_optional="all"
            )

        self.flux_points = fpe.run(self.datasets)

        self._flux_points_dataset = FluxPointsDataset(
            data=self.flux_points, models=self.datasets.models
        )

    def change_model(self, model, optimize=False, **kwargs):
        """
        To change a target model

        Args:
            model (str or gammapy.modeling.models.SpectralModel): a new target model
            optimize (bool): perform optimization (JointAnalysis.optimize)
                Default: False
            **kwargs: passed to JointAnalysis.optimize
        """

        if self._num_of_models == 1:
            prevmodel = self.target_model[0].spectral_model.tag[0]
        else:
            prevmodel = "Composite model"

        if type(model) == str:
            spectral_model = default_model(model, **kwargs)
            if model is None:
                self._logging.error("The input model is not supported yet.")
                return
        elif model == SkyModel:
            spectral_model = model.spectral_model
            self._logging.info(f"A model, {spectral_model.tag[0]}, is imported")
        elif hasattr(model, "tag"):
            spectral_model = model
            self._logging.info(f"A model, {spectral_model.tag[0]}, is imported")

    
        for model in self._find_target_model():
            self.datasets.models[model.name].spectral_model = spectral_model

        if optimize:
            self.optimize(**kwargs)
            self._model_change_flag = False
        else:
            self._model_change_flag = True

        newmodel = spectral_model.tag[0]
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
                    spectral_model = gammapy_model.PowerLawSpectralModel()
                    self._logging.warning(f"The model is assumed to be a power law.")
                elif type(model) == str:
                    spectral_model = default_model(model)
                elif hasattr(model, "tag"):
                    spectral_model = model

                target_name = kwargs.pop("target_name", "new component")
                model = SkyModel(spectral_model=spectral_model, spatial_model = self._target_spatial_model, name=target_name)
            name = kwargs.pop("name", f"dataset_{len(self.datasets)}")
            new_dataset = FluxPointsDataset(data=data, models=model, name=name, **kwargs)
            
        else:
            self._logging.error(f"This data type is not supported yet.")

        if not(sync):
            fit_ = Fit()
            fit_results = fit_.run(new_dataset)
            
            if fit_results.success:
                self.datasets.append(new_dataset)
                self._logging.info("A new dataset is successfully added.")
            else:
                self._logging.error("A new dataset is NOT added.")
                self._logging.error("Initial parameters for the new dataset model may not be proper.")
                emin_d = new_dataset.data.energy_min[0]
                emax_d = new_dataset.data.energy_max[-1]
                self.plot(spectral_model, energy_bounds = [emin_d, emax_d])
                self.plot_flux(new_dataset)
        else:
            self.datasets.append(new_dataset)
            self._logging.info("A new dataset is successfully added.")
                
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
            self.fermi.analysis("sed")

        if veritas and not(hasattr(self.veritas, "_flux_points_dataset")):
            self.veritas.analysis("sed")

        if joint and not(self._fit_flag):
            fit = False
        else:
            fit = True

        cmap = plt.get_cmap("tab10")
        i = 0

        if veritas:
            vts = self.veritas._flux_points_dataset

            if show_flux_points:
                self.plot_flux(vts, color = cmap(i), label="VERITAS")

            if not(fit) and show_model:
                veritas_model = vts.models[0].spectral_model
                self.plot_model(veritas_model, band=False, energy_bounds=vts._energy_bounds, color=cmap(i))
            i+=1

        if fermi:
            fermi = self._fermi_flux_dataset()
            
            if show_flux_points:
                self.plot_flux(fermi, color = cmap(i), label="Fermi-LAT")

            if not(fit) and show_model:
                fermi_model = fermi.models[0].spectral_model
                self.plot_model(fermi_model, band=False, energy_bounds=fermi._energy_bounds, color=cmap(i))
            i+=1

        emin = 1*u.TeV
        emax = 1*u.TeV

        for dataset in self.datasets:
            if type(dataset) == FluxPointsDataset:
                emin_d = dataset.data.energy_min[0]
                emax_d = dataset.data.energy_max[-1]
            else:
                emin_d = dataset.energy_range_total[0]
                emax_d = dataset.energy_range_total[1]
            
            emin = min(emin, emin_d/2.)
            emax = max(emax, emax_d*2.)

            if (dataset.name != "fermi") and (dataset.name != "veritas"):
                self.plot_flux(dataset, color=cmap(i))
                other_model = dataset.models[0].spectral_model
                self.plot_model(other_model, energy_bounds=[emin, emax], color=cmap(i))
                i+=1
            
        self._energy_bounds = [emin, emax]

        if joint and show_model:
            for model in np.atleast_1d(self.target_model):
                if fit:
                    self.plot_model(model.spectral_model, band=True, energy_bounds=self._energy_bounds, color=cmap(i))
                else:
                    self.plot_model(model.spectral_model, band=False, energy_bounds=self._energy_bounds, color=cmap(i), ls="--")
            i+=1

        plt.xlim(emin, emax)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(kwargs.get("ylim_l", 1e-13),kwargs.get("ylim_h", 1e-8))
        plt.legend(fontsize=13)
        plt.grid(which="major", ls="-")
        plt.grid(which="minor", ls=":")
        plt.xlabel("Energy [TeV]", fontsize=13)
        plt.ylabel(r"Energy flux [erg/cm$^2$/s]", fontsize=13)

    @staticmethod
    def plot_flux(dataset, **kwargs):
        if type(dataset) == FluxPointsDataset:
            label = kwargs.pop("label", dataset.name)
            dataset.data.plot(label=label, energy_power=0, sed_type="e2dnde", **kwargs)

    @staticmethod
    def plot_model(output, band=False, **kwargs):
        if type(output) == SkyModel:
            output = output.spectral_model

        if hasattr(output, "plot"):
            energy_bounds = kwargs.pop("energy_bounds", [0.1*u.GeV, 50*u.TeV])
            output.plot(sed_type="e2dnde", energy_bounds=energy_bounds, **kwargs)
            if band:
                face_color = kwargs.get("color", "k")
                output.plot_error(energy_bounds=energy_bounds,
                                         sed_type="e2dnde", alpha=0.2, facecolor=face_color)


    def construct_dataset(self):
        self.datasets = Datasets([self.fermi.datasets, self.veritas.stacked_dataset])
        possible_models = self._find_target_model()
        self._num_of_models = len(possible_models)
        self.veritas.stacked_dataset.models = possible_models


    def _find_target_model(self):
        target_pos = self.fermi.datasets.models[self.target_name[1]].spatial_model.position
        th2cut = self.veritas._on_region.radius.value

        models = []
        for model in self.fermi.datasets.models:
            if model.name != 'galdiff' and model.name != 'isodiff':
                if target_pos.separation(model.spatial_model.position).deg < th2cut:
                    models.append(model)
        return models

    def _fermi_flux_dataset(self, models=None, ul_ts_threshold = 9, **kwargs):

        if models is None:
            if self.target_name[0] in self.fermi.datasets.models.names:
                models = self.fermi.datasets.models[self.target_name[0]]
            else:
                models = self.fermi.datasets.models[self.target_name[1]]

        fermi_sed = kwargs.pop("fermi_sed", f"{self.fermi._outdir}/{self._fermi_state}_sed.fits")

        if not(os.path.isfile(fermi_sed)):
            self.fermi.analysis("sed", state_file = self._fermi_state)

        table = Table.read(fermi_sed, format='fits', hdu=1)
        table["is_ul"] = table["ts"]<ul_ts_threshold
        data = FluxPoints.from_table(table, sed_type="likelihood")

        fermi_dataset = FluxPointsDataset(data=data, models=models)

        fermi_dataset.mask_safe = ~fermi_dataset.data.to_table()["is_ul"]
        fermi_dataset.mask_fit = ~fermi_dataset.data.to_table()["is_ul"]

        return fermi_dataset