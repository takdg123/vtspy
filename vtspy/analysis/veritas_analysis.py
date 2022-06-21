import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import copy
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time

from ..utils import logger, LiMaSiginficance
from .. import utils
from ..model import default_model
from ..plotting import plot_ROI
from ..config import JointConfig

from gammapy.data import DataStore
from gammapy.datasets import Datasets, SpectrumDataset, FluxPointsDataset

from gammapy.maps import WcsGeom, MapAxis, RegionGeom

from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)

from gammapy.modeling import Fit

import gammapy.modeling.models as gammapy_model

from gammapy.modeling.models import SkyModel

from regions import CircleSkyRegion

from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator, FluxPoints, LightCurveEstimator



class VeritasAnalysis:

	_energy_axis = MapAxis.from_energy_bounds(
		    0.05, 50., nbin=100, per_decade=False, unit="TeV", name="energy"
		)
	_energy_axis_true = MapAxis.from_energy_bounds(
		    0.05, 50., nbin=200, per_decade=False, unit="TeV", name="energy_true"
		)
	_excluded_regions = []

	def __init__(self, state_file = "initial", config_file='config.yaml', overwrite=False, verbosity=1, **kwargs):
		"""
	    This is to perform a simple VERITAS analysis and to prepare the joint-fit
	    analysis. All analysis is done with the gammapy package
	    
	    Args:
	        state_file (str): state filename (pickle)
	        	Default: initial
	        config_file (str): config filename (yaml)
	            Default: config.yaml
	        overwrite (bool): overwrite the state
	            Default: False
	        verbosity (int)
	        **kwargs: passed to VeritasAnalysis.setup
	    """
		self._verbosity = verbosity
		
		self.config = JointConfig.get_config(config_file).pop("veritas")

		self._logging = logger(self.verbosity)

		self._logging.info("Initialize the VERITAS analysis.")

		self._outdir = self.config['fileio']['outdir']
		
		self._eff_cut = self.config['cuts']['eff_cut']
		self._bias_cut = self.config['cuts']['bias_cut']
		self._max_region_number = self.config['selection']['max_region_number']

		self._energy_bins = MapAxis.from_bounds(0.1, 10, nbin=8, interp="log", unit="TeV").edges

		if overwrite or not(os.path.isfile(f"./{self._outdir}/initial.pickle")):

			self.setup(**kwargs)

			self._logging.debug("Save the initial state.")
			
			self.save_state("initial", init=True)

			self._logging.info("The initial setup is saved [state_file = initial].")

		else:
			self._logging.info("The setup is found [state_file = {}]. Read the state.".format(state_file))
			flag = self.load_state(state_file)
			
			if flag == -1:
				return

		self._logging.info("Completed (VERITAS initialization).")

	@property
	def target(self):
		"""
        Return:
            astropy.SkyCoord
        """
		return SkyCoord(ra = self.config["selection"]["ra"], 
						dec = self.config["selection"]["dec"], 
                  		unit="deg", frame="icrs")

	@property
	def target_name(self):
		"""
        Return:
            str: target name
        """
		return self.config["selection"]["target"]

	@property
	def obs_ids(self):
		"""
        Return:
            list: list of observation id
        """
		return self._obs_ids
	
	@property
	def energy_bins(self):
		"""
        Return:
        	MapAxis.edges: energy bins used in flux points.
        """
		return self._energy_bins

	@property
	def verbosity(self):
		"""
        Return:
        	int
        """
		return self._verbosity


	
	def print_flux(self):
		"""
        Return:
        	astropy.table: flux points in SED
        """
		if hasattr(self, "flux_points"):
			return self.flux_points.to_table(sed_type="e2dnde", formatted=True)

	def print_lightcurve(self, sed_type='eflux'):
		"""
        Return:
        	astropy.table: flux points in lightcurve
        """
		if hasattr(self, "_lightcurve"):
			return self._lightcurve.to_table(sed_type='eflux',format='lightcurve')
	
	def print_models(self):
		"""
        Return:
        	astropy.table: fit parameters
        """
		if hasattr(self, "stacked_dataset"):
			return self.stacked_dataset.models.to_parameters_table()

	def peek_dataset(self):
		"""
        Show dataset information
        """
		self.stacked_dataset.peek()

	def save_state(self, state_file, init=False):
		"""
		Save the state

		Args:
		    state_file (str): the name of state
		    init (bool): check whether this is the initial analysis.
		        Default: False
		"""

		if (init==False) and (state_file == "initial"):
		    self._logging.warning("The 'inital' state is overwritten. This is not recommended.")
		    self._logging.warning("The original 'inital' state is archived in the '_orig' folder.")
		    os.system(f"mkdir ./{self._outdir}/_orig")
		    for file in glob.glob(f"./{self._outdir}/*initial*"):
		        os.sytem(f"mv {file} ./{self._outdir}/_orig/")

		filename = f"./{self._outdir}/{state_file}.pickle".format(state_file)
		with open(filename, 'wb') as file:
			del(self._logging)
			pickle.dump(self, file)
			self._logging = logger(self.verbosity)

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
		except:
			self._logging.error("The state file does not exist. Check the name again")
			return -1

	

	def setup(self, **kwargs):
		"""
	    This is to initialize the VERITAS analysis; e.g., construct datasets
	    To change the setting for this setup, check config file.

	    VeritasAnalysis.config
	    
	    Args:
	        **kwargs: passed to VeritasAnalysis.construct_dataset
	    """

		selection = dict(
		    type="sky_circle",
		    frame="icrs",
		    lon= self.target.ra,
		    lat= self.target.dec,
		    radius = self.config["selection"]['radius'] * u.deg,
		)

		self._logging.info("Load the data files.")

		if not(os.path.exists(f"{self._outdir}/hdu-index.fits.gz")):
			self._logging.warning("The 'hdu index' file is not found.")
			try:
				from pyV2DL3 import generateObsHduIndex
			except:
				self._logging.error("The pyV2DL3 package is required to proceed.")
				return

			import glob
			filelist = glob.glob(f"{self._outdir}/*fit*")
			generateObsHduIndex.create_obs_hdu_index_file(filelist, index_file_dir=self._outdir)
			self._logging.info("The hdu-index and obs-index files are created.")	

		self._data_store = DataStore.from_dir(f"{self._outdir}")

		self._obs_ids = self._data_store.obs_table.select_observations(selection)["OBS_ID"]

		self.observations = self._data_store.get_observations(self._obs_ids, required_irf=["aeff", "edisp"])
		time_intervals = [self.config["selection"]["tmin"], self.config["selection"]["tmax"]]
		self.observations, self._obs_ids = utils.time_filter(self.observations, time_intervals, time_format="mjd")
		self._logging.info(f"The number of observations is {len(self.observations)}")

		self._logging.info("Define exclusion regions.")
		self._exclusion_mask = self._exclusion_from_bright_srcs(**kwargs)
		self.add_exclusion_region(coord=[self.target.ra, self.target.dec], radius=self.config["selection"]["exc_on_region_radius"])
		
		self._logging.info("Define ON- and OFF-regions.")
		th2cut = self.config["cuts"]['th2cut']
		self._on_region = CircleSkyRegion(center=self.target, radius=Angle(np.sqrt(th2cut)*u.deg))
		
		self.construct_dataset(**kwargs)

	def fit(self, model = "PowerLaw", state_file="simple", save_state=True, **kwargs):
		"""
        Perform a simple fitting with a given model: 
        PowerLaw, LogParabola, ...
        
        Args:
            model (str or gammapy.models): model name or function
                Default: "PowerLaw"
            state_file (str): state filename (pickle)
	        	Default: simple
	        save_state(bool)
	        	Default: True
	        **kwargs: passed to vtspy.model.default_model
        """

		if type(model) == str:
			spectral_model = default_model(model, **kwargs)
		elif hasattr(model, "tag"):
			spectral_model = model

		spectral_model = SkyModel(spectral_model=spectral_model, name=self.target_name)
		self.datasets.models = spectral_model
		self.stacked_dataset.models = [spectral_model]

		fit_joint = Fit()
		self.fit_results = fit_joint.run(datasets=[self.stacked_dataset])

		if self.fit_results.success:
			self._logging.info("Fit successfully.")
			if save_state:
				self.save_state(state_file)
		else:
			self._logging.error("Fit failed.")

	def analysis(self, jobs=["sed"], state_file="analyzed", **kwargs):
		"""
		Perform a simple analysis, e.g., SED, lightcurve

		Args:
			jobs (list): list of jobs, 'sed', and/or 'lc'.
				Default: ['sed']
			state_file (str): state filename (pickle)
				Default: analyzed
			**kwargs: passed to vtspy.utils.define_time_intervals
		"""

		if type(jobs) == str:
			jobs = [jobs]

		if "sed" in jobs:
			self._logging.info("Generating flux points and SED...")
			self._energy_bins = kwargs.get("energy_bins", self._energy_bins)

			fpe = FluxPointsEstimator(
			    energy_edges=self.energy_bins, 
			    source=self.target_name, selection_optional="all", **kwargs
			    )

			self.flux_points = fpe.run(datasets=[self.stacked_dataset])
			

			self._flux_points_dataset = FluxPointsDataset(
			    data=self.flux_points, models=self.stacked_dataset.models
			)
			self._logging.info("Completed.")

		if "lc" in jobs:
			self._logging.info("Generating lightcurve...")

			emin = kwargs.pop("emin", self.config["selection"]['emin'])
			emax = kwargs.pop("emax", self.config["selection"]['emax'])
			ul = kwargs.pop("ul", 2)
			tmin = kwargs.pop("tmin", self.config["selection"]['tmin'])
			tmax = kwargs.pop("tmax", self.config["selection"]['tmax'])

			time_intervals = utils.define_time_intervals(tmin, tmax, **kwargs)
			self._logging.info(f"The number of time intervals is {len(time_intervals)}")
			
			lc_maker = LightCurveEstimator(
			    energy_edges=[emin, emax] * u.TeV,
			    source=self.target_name,
			    time_intervals=time_intervals,
			    selection_optional="all"
			)

			self._lightcurve = lc_maker.run(self.datasets)
			self._lightcurve.sqrt_ts_threshold_ul = ul
			self._logging.info("Generating lightcurve is completed.")

		self.save_state(state_file)


	def plot(self, output, **kwargs):
		"""
        Show various results: fit result, flux, SED, and lightcurve
        
        Args:
            output (str): a plot to show
                Options: ["roi", "fit", "flux", "sed", "lc"]
            filename (str): read the output (from FermiAnalysis.analysis)
        """
		if output == "roi":
			plot_ROI(veritas = self)
		elif output == "fit":
			self.stacked_dataset.plot_fit()
		elif output == "flux":
			ax = plt.gca()
			self.flux_points.plot(ax, sed_type="e2dnde", color="lightblue", label=self.target_name)
			self.flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde");
			ax.legend()
		elif output == "sed":
			kwargs_spectrum = {**kwargs, "kwargs_model": {"color":"blue", "label":"Pwl"}, "kwargs_fp":{"color":"blue", "marker":"o", "label":self.target_name}}
			kwargs_residuals = {"color": "blue", "markersize":4, "marker":'s', }
			ax_spec, ax_res = self._flux_points_dataset.plot_fit(kwargs_spectrum=kwargs_spectrum)
		elif output == "lc":
			self._lightcurve.plot(sed_type='eflux', label=self.target_name)

	def add_exclusion_region(self, coord = None, name=None, radius=0.3, update_dataset=False, **kwargs):
		"""
        Add exclusion region manually
        
        Args:
            coord (list, optioanl): [ra, dec]
            name (list, optional): [ra, dec]
            radius (float): size of an exlusion region
            update_dataset (bool): update dataset with the new exclusion region
        """
		if coord is not None:
			self._excluded_regions.append(CircleSkyRegion(
				center=SkyCoord(coord[0], coord[1], unit="deg", frame="icrs"),
				radius=radius * u.deg,))
		elif name is not None:
			src = SkyCoord.from_name(name)
			self._excluded_regions.append(CircleSkyRegion(
				center=SkyCoord(src.ra, src.dec, unit="deg", frame="icrs"),
				radius=radius * u.deg,))
		else:
			self._logging.error("Either coord or name should be provided.")
			return

		geom = WcsGeom.create(
		    	npix=(150, 150), binsz=0.05, skydir=self.target.galactic, 
		    	proj="TAN", frame="icrs"
		)

		self._exclusion_mask = geom.region_mask(regions=self._excluded_regions, inside=False)

		if update_dataset:
			self._on_region = CircleSkyRegion(center=self.target, radius=Angle(np.sqrt(th2cut)*u.deg))
			self.construct_dataset(**kwargs)

	def construct_dataset(self, **kwargs):
		"""
        Construct dataset for the gammapy analysis

        Args:
        	**kwargs: e.g., eff_cut, bias_cut, max_region_number, or others
        """

		self._eff_cut = kwargs.pop("eff_cut", self._eff_cut)
		self._bias_cut = kwargs.pop("bias_cut", self._bias_cut)
		self._max_region_number = kwargs.pop("max_region_number", self._max_region_number)

		datasets = Datasets()

		dataset_maker = SpectrumDatasetMaker(selection=["counts", "exposure", "edisp"], containment_correction=False)
		bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=self._exclusion_mask, max_region_number=self._max_region_number)
		
		if self._eff_cut > 0:
			safe_mask_maker_eff = SafeMaskMaker(methods=["aeff-max"], aeff_percent=self._eff_cut)
		else:
			safe_mask_maker_eff = None

		if self._bias_cut > 0:
			safe_mask_maker_bias = SafeMaskMaker(methods=["edisp-bias"], bias_percent=self._bias_cut)
		else:
			safe_mask_maker_bias = None

		geom = RegionGeom.create(region=self._on_region, axes=[self._energy_axis])
		dataset_empty = SpectrumDataset.create(
    		geom=geom, energy_axis_true=self._energy_axis_true
		)


		for obs_id, observation in zip(self._obs_ids, self.observations):
			dataset = dataset_maker.run(
				dataset_empty.copy(name=str(obs_id)), observation
			)

			dataset_on_off = bkg_maker.run(dataset, observation)

			if safe_mask_maker_eff is not None:
				dataset_on_off = safe_mask_maker_eff.run(dataset_on_off, observation)

			if safe_mask_maker_bias is not None:
				dataset_on_off = safe_mask_maker_bias.run(dataset_on_off, observation)

			datasets.append(dataset_on_off)

		self.datasets = datasets
		self.info_table = self.datasets.info_table()
		self.stacked_dataset = self.datasets.stack_reduce(name="veritas")
		self._N_off = sum(self.info_table["counts_off"])
		self._N_on = sum(self.info_table["counts"])
		self._alpha = np.average(self.info_table["alpha"], weights=1/self.info_table["livetime"])
		self._sigma = LiMaSiginficance(self._N_on, self._N_off, self._alpha)
		self._logging.info(r"N_on: {}, N_off: {}, alpha: {:.3f}, and sigma={:.1f}".format(self._N_on, self._N_off, self._alpha, self._sigma))

	def _exclusion_from_bright_srcs(self):
		srcfile = self.config["background"]["file"]
		distance = self.config["background"]["distance"]
		magnitude = self.config["background"]["magnitude"]
		ex_radius = self.config["selection"]["exc_radius"]

		bright_sources = utils.bright_source_list(srcfile)

		roi_cut = (abs(bright_sources[:,0]-self.target.ra.deg) < distance) \
        		* (abs(bright_sources[:,1]-self.target.dec.deg) < distance) \
        		* (bright_sources[:,2]+bright_sources[:,3] < magnitude)
                                                                                             
		bright_sources = bright_sources[roi_cut]
        
		for src_pos in bright_sources:
			self._excluded_regions.append(CircleSkyRegion(
				center=SkyCoord(src_pos[0], src_pos[1], unit="deg", frame="icrs"),
				radius=ex_radius * u.deg,))

		geom = WcsGeom.create(
			npix=(150, 150), binsz=0.05, skydir=self.target.galactic,
			proj="TAN", frame="icrs"
		)

		return geom.region_mask(regions=self._excluded_regions, inside=False)

