
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle


from ..utils import logger, bright_source_list
from ..config import GammaConfig
from ..plotting import veritas_plotter

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

from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator, FluxPoints



class VeritasAnalysis:
	def __init__(self, state_file = "initial", config_file='config.yaml', overwrite=False, verbosity=1, **kwargs):
		self._verbosity = verbosity
		self.config = GammaConfig(config_file=config_file, verbosity=(self.verbosity-1)).config

		self._logging = logger(self.verbosity)

		self._logging.info("Initialize the VERITAS analysis.")

		self._outdir = self.config['fileio_vtspy']['veritas']
		
		self._excluded_regions = []
		
		self._energy_axis = MapAxis.from_energy_bounds(
		    0.05, 50., nbin=100, per_decade=False, unit="TeV", name="energy"
		)

		self._energy_axis_true = MapAxis.from_energy_bounds(
		    0.05, 50., nbin=200, per_decade=False, unit="TeV", name="energy_true"
		)

		
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

			self._exclusion_mask = self._exclusion_from_bright_srcs(**kwargs)
			self.add_exclusion_region(coord=[self.target.ra, self.target.dec], radius=0.7)
			th2cut = kwargs.get("th2cut", 0.008)
			self.on_region = CircleSkyRegion(center=self.target, radius=Angle(np.sqrt(th2cut)*u.deg))

		self._logging.info("Completed (VERITAS initialization).")
	


	@property
	def target(self):
		return SkyCoord(ra = self.config["selection"]["ra"], 
						dec = self.config["selection"]["dec"], 
                  		unit="deg", frame="icrs")

	@property
	def target_name(self):
		return self.config["selection"]["target"]
	
	@property
	def verbosity(self):
		return self._verbosity

	@property
	def print_flux(self):
		if hasattr(self, flux_points):
			return self.flux_points.to_table(sed_type="e2dnde", formatted=True)
	
	@property
	def print_parms(self):
		if hasattr(self, stacked_dataset):
			return self.stacked_dataset.models.to_parameters_table()

	def save_state(self, state_file, init=False):
		"""
		Save the state

		Args:
		    state_file (str): passed to fermipy.write_roi
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
		state_file (str): passed to fermipy.write_roi
		"""
		filename = f"./{self._outdir}/{state_file}.pickle".format(state_file)
		with open(filename, 'rb') as file:
			self.__dict__.update(pickle.load(file).__dict__)
			
	
	def peek_dataset(self):
		self.stacked_dataset.peek()

	def setup(self, radius = 2.0, max_region_number = 6, th2cut=0.008, **kwargs):
		selection = dict(
		    type="sky_circle",
		    frame="icrs",
		    lon=self.target.ra,
		    lat=self.target.dec,
		    radius = radius * u.deg,
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

		self._observations = self._data_store.get_observations(self._obs_ids, required_irf=["aeff", "edisp"])

		self._logging.info("Define exclusion regions.")
		self._exclusion_mask = self._exclusion_from_bright_srcs(**kwargs)
		self.add_exclusion_region(coord=[self.target.ra, self.target.dec], radius=0.7)
		
		self._logging.info("Define ON- and OFF-regions.")
		self._on_region = CircleSkyRegion(center=self.target, radius=Angle(np.sqrt(th2cut)*u.deg))
		
		self.create_dataset(**kwargs)

		self.stacked_dataset = self.datasets.stack_reduce()

	def obs_time_filter(self, t_start, t_end, time_format="mjd"):
		time_interval= Time([str(t_start),str(t_end)], format=time_format, scale="utc")
		short_observations = self._observations
		self._observations = short_observations.select_time(time_interval)
		self._obs_ids = short_observations.ids
		self._logging.info(f"Number of observations after time filtering: {len(short_observations)}\n")

	def simple_fit(self, model = "PowerLaw"):

		if type(model) == str:
			if model == "PowerLaw":
				spectral_model = gammapy_model.PowerLawSpectralModel(
				    amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
				    index=2.5,
				    reference=1 * u.TeV,
				)
			elif model == "LogParabola":
				spectral_model = gammapy_model.LogParabolaSpectralModel(
				     alpha=3,
				     amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
				     reference=1 * u.TeV,
				     beta=2,
				)
		else:
			spectral_model = model

		spectral_model = SkyModel(spectral_model=spectral_model, name=self.target_name)

		self.stacked_dataset.models = [spectral_model]

		fit_joint = Fit()
		self.fit_results = fit_joint.run(datasets=[self.stacked_dataset])
		if self.fit_results.success:
			self._logging.info("Fit successfully.")
		else:
			self._logging.error("Fit fails.")

	def simple_analysis(self, jobs=["flux", "sed"], **kwargs):

		if "flux" in jobs:
			energy_bins = kwargs.get("energy_bins", np.geomspace(0.1, 10, 12) * u.TeV)

			fpe = FluxPointsEstimator(
			    energy_edges=energy_bins, 
			    source=self.target_name, selection_optional="all", **kwargs
			    )

			self.flux_points = fpe.run(datasets=[self.stacked_dataset])
		
		if "sed" in jobs:
			self._flux_points_dataset = FluxPointsDataset(
			    data=self.flux_points, models=self.stacked_dataset.models
			)

	def plotting(self, plots):
		if plots == "fit":
			veritas_plotter(plots, self.stacked_dataset)
		elif plots == "flux":
			veritas_plotter(plots, self.flux_points)
		elif plots == "sed":
			veritas_plotter(plots, self._flux_points_dataset)

	def _exclusion_from_bright_srcs(self, srcfile="Hipparcos_MAG8_1997", distance = 1.75, ex_radius=0.25, magnitude=7):
		bright_sources = bright_source_list(srcfile)

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

	def add_exclusion_region(self, coord = None, name=None, radius=0.3, **kwargs):
		if coord is not None:
			self._excluded_regions.append(CircleSkyRegion(
				center=SkyCoord(coord[0], coord[1], unit="deg", frame="icrs"),
				radius=radius * u.deg,))
		elif name is not None:
			src = SkyCoord.from_name(name)
			self._excluded_regions.append(CircleSkyRegion(
				center=SkyCoord(src.ra, src.dec, unit="deg", frame="icrs"),
				radius=radius * u.deg,))

		geom = WcsGeom.create(
		    	npix=(150, 150), binsz=0.05, skydir=self.target.galactic, 
		    	proj="TAN", frame="icrs"
		)

		self._exclusion_mask = geom.region_mask(regions=self._excluded_regions, inside=False)

	def create_dataset(self, max_region_number = 6, eff_cut = 0, bias_cut = 0, **kwargs):
		datasets = Datasets()

		dataset_maker = SpectrumDatasetMaker(selection=["counts", "exposure", "edisp"], containment_correction=False)
		bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=self._exclusion_mask, max_region_number=max_region_number)
		
		if eff_cut > 0:
			safe_mask_maker_eff = SafeMaskMaker(methods=["aeff-max"], aeff_percent=eff_cut*100)
		else:
			safe_mask_maker_eff = None
		
		if bias_cut > 0:
			safe_mask_maker_bias = SafeMaskMaker(methods=["edisp-bias"], bias_percent=bias_cut*100)
		else:
			safe_mask_maker_bias = None

		geom = RegionGeom.create(region=self._on_region, axes=[self._energy_axis])
		dataset_empty = SpectrumDataset.create(
    		geom=geom, energy_axis_true=self._energy_axis_true
		)

		for obs_id, observation in zip(self._obs_ids, self._observations):
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