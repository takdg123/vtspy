import os
import glob

import yaml

import uproot

from astropy.io import fits

from . import utils
from .utils import logger, SCRIPT_DIR

from pathlib import Path

class JointConfig:
	"""
	This is to generate the configuration file compatible to
	the Fermipy configuration. The input file is the VERITAS
	event file from EventDisplay. The format can be either
	`.root' or `.fits'.

	Args:
	    files (str): a input file or directory containing
	    	root or fits files (root or fits)
	    config_file (str): Fermi config filename (yaml)
	    	Default: config.yaml
	    verbosity (int)
	    **kwargs: passed to JointConfig.init
	"""

	def __init__(self, files=None, config_file="config.yaml", verbosity=1, **kwargs):
		self._logging = logger(verbosity=verbosity)

		self._filename = config_file
		path = Path(config_file)
		if path.is_file() and (files is None):
			config = self.get_config(config_file)
			self.fermi_config = config.pop("fermi")
			self.veritas_config = config.pop("veritas")
			self.print_info()
			self._logging.info(f'a configuration file ({config_file}) is loaded.')
		else:
			self.init(files=files, config_file=config_file, **kwargs)
			self.print_info(config_file=config_file)
			self._logging.info(f'a configuration file ({config_file}) is created.')

	def init(self, files=None, config_file="config.yaml", **kwargs):
		"""
		Generate a default configuration file
		Args:
			files (str): a input file or directory containing
		    	root or fits files (root or fits)
		    config_file (str): Fermi config filename (yaml)
		    	Default: config.yaml
			fermi_outdir (str): path to the Fermi-LAT output directory
		        Default: ./fermi/
		    fermi_data (str): path to the Fermi-LAT data directory
	        	Default: ./fermi/
	        veritas_outdir (str): path to the VERITAS output directory
	        	Default: ./veritas/
			veritas_data (str): path to the VERITAS data directory
	        	Default: ./veritas/
	        joint_outdir (str): path to the joint-fit output directory
	        	Default: ./joint/
	        **kwargs
        """

		fermi_outdir = kwargs.pop("fermi_outdir", "./fermi/")
		fermi_data = kwargs.pop("fermi_data", "./fermi/")
		veritas_outdir = kwargs.pop("veritas_outdir", "./veritas/")
		veritas_data = kwargs.pop("veritas_data", "./veritas/")
		joint_outdir = kwargs.pop("joint_outdir", "./joint/")

		gald = kwargs.pop("gald", "gll_iem_v07.fits")
		iso = kwargs.pop("iso", "iso_P8R3_SOURCE_V3_v1.txt")
		info = kwargs.pop("info", {})

		if files is None:
			filelist = glob.glob(veritas_data+"/*")
		else:
			filelist = glob.glob(files+"/*")

		for file in filelist:
			if ".gz" in file:
				filelist.remove(file)

		self.fermi_config = self._empty4fermi(outdir = fermi_outdir, datadir = fermi_data, gald=gald, iso=iso)
		self.veritas_config = self._empty4veritas(outdir = veritas_outdir, datadir = veritas_data)

		self.veritas_config["data"]["anasum"] = os.path.dirname(filelist[0])

		info = {**info,
			'fermi':{
			'selection':{
				'ra': None,
				'dec': None,
				'tmin': None,
				'tmax':None}},
			'veritas':{
			'selection':{
				'ra': None,
				'dec': None,
				'tmin': None,
				'tmax': None,
			}
			}
		}

		ra = info['fermi']['selection']['ra']
		dec = info['fermi']['selection']['dec']

		for file in filelist:

			if file!=None:

				if '.root' in file:

					anasum = uproot.open(file)

					tRun = anasum['total_1']['stereo']['tRunSummary']

					if info['fermi']['selection']['ra'] == None:
						ra = float(tRun['TargetRAJ2000'].arrays(library="np")['TargetRAJ2000'][0])
					else:
						temp = float(tRun['TargetRAJ2000'].arrays(library="np")['TargetRAJ2000'][0])
						if temp != ra:
							self._logging.error("[Error] RA values in input files are different.")

					if dec == None:
						dec = float(tRun['TargetDecJ2000'].arrays(library="np")['TargetDecJ2000'][0])
					else:
						temp = float(tRun['TargetDecJ2000'].arrays(library="np")['TargetDecJ2000'][0])
						if temp != dec:
							self._logging.error("[Error] DEC values in input files are different.")

					tmin_mjd = tRun['MJDOn'].arrays(library="np")['MJDOn'][0]
					tmin_utc = utils.MJD2UTC(tmin_mjd)
					tmin = utils.UTC2MET(tmin_utc[:10])

					tmax_mjd = tRun['MJDOff'].arrays(library="np")['MJDOff'][0]
					tmax_utc = utils.MJD2UTC(tmax_mjd)
					tmax = utils.UTC2MET(tmax_utc[:10])+60*60*24

				elif 'anasum.fits' in file:

					header = fits.open(file)[1].header
					if ra == None:
						ra = header['RA_OBJ']
					else:
						temp = header['RA_OBJ']
						if temp != ra:
							self._logging.error("[Error] RA values in input files are different.")

					if dec == None:
						dec = header['DEC_OBJ']
					else:
						temp = header['DEC_OBJ']
						if temp != dec:
							self._logging.error("[Error] DEC values in input files are different.")

					tmin_utc = header['DATE-OBS']
					tmin = utils.UTC2MET(tmin_utc[:10])
					tmin_mjd = utils.UTC2MJD(tmin_utc)

					tmax_utc = header['DATE-END']
					tmax = utils.UTC2MET(tmax_utc[:10])+60*60*24
					tmax_mjd = utils.UTC2MJD(tmax_utc)

					target = header['OBJECT']
				else:
					continue

				if info['fermi']['selection']['tmin'] is not None:
					if info['fermi']['selection']['tmin'] < tmin:
						tmin = info['fermi']['selection']['tmin']
					else:
						info['fermi']['selection']['tmin'] = tmin

				if info['fermi']['selection']['tmax'] is not None:
					if info['fermi']['selection']['tmax'] > tmax:
						tmax = info['fermi']['selection']['tmax']
					else:
						info['fermi']['selection']['tmax'] = tmax

				if info['veritas']['selection']['tmin'] is not None:
					if info['veritas']['selection']['tmin'] < tmin_mjd:
						tmin_mjd = info['veritas']['selection']['tmin']
					else:
						info['veritas']['selection']['tmin'] = tmin_mjd
				if info['veritas']['selection']['tmax'] is not None:
					if info['veritas']['selection']['tmax'] > tmax_mjd:
						tmax_mjd = info['veritas']['selection']['tmax']
					else:
						info['veritas']['selection']['tmax'] = tmax_mjd


				info = {**info,
					'fermi':{
					'selection':{
						'ra': ra,
						'dec': dec,
						'tmin': tmin,
						'tmax': tmax,
						'target': target},
						},
					'veritas':{
					'selection':{
						'ra': ra,
						'dec': dec,
						'tmin': tmin_mjd,
						'tmax': tmax_mjd,
						'target': target,
					}}
					}



		info['fermi'] = self._filter(self.fermi_config, info['fermi'])
		info['veritas'] = self._filter(self.veritas_config, info['veritas'])

		self.fermi_config = self._update(self.fermi_config, info['fermi'])
		self.veritas_config = self._update(self.veritas_config, info['veritas'])
		self.joint_config = {'fileio':{
								'outdir': joint_outdir,
							}}
		info = {"fermi": self.fermi_config, "veritas": self.veritas_config, "joint": self.joint_config}

		self.set_config(info, config_file)

		self.config = info

	def change_time_interval(self, tmin, tmax, scale = "utc", instrument="all"):
		"""
		Change and update a time interval

		Args:
		tmin (float or str): start time
		tmax (float or str): end time
		scale (str): "utc", "mjd", or "met"
			Default: "utc"
		instrument (str): "fermi", "veritas", or "all"
			Default: "all"


		"""

		if scale.lower() == "utc":
			tmin_mjd = utils.UTC2MJD(tmin)
			tmax_mjd = utils.UTC2MJD(tmax)
			tmin_met= utils.UTC2MET(tmin[:10])
			tmax_met = utils.UTC2MET(tmax[:10])+60*60*24
		elif scale.lower() == "mjd":
			tmin_mjd = tmin
			tmax_mjd = tmax
			tmin_utc = utils.MJD2UTC(tmin)
			tmax_utc = utils.MJD2UTC(tmax)
			tmin_met = utils.UTC2MET(tmin_utc[:10])
			tmax_met = utils.UTC2MET(tmax_utc[:10])+60*60*24
		elif scale.lower() == "met":
			tmin_met = tmin
			tmax_met = tmax
			tmin_utc = utils.MET2UTC(tmin)
			tmax_utc = utils.MET2UTC(tmax)
			tmin_mjd = utils.UTC2MJD(tmin_utc)
			tmax_mjd = utils.UTC2MJD(tmax_utc)
		else:
			self._logging.error("The input 'scale' parameter is not 'MJD', 'MET', or 'UTC'.")
			return

		if instrument.lower() == "fermi" or instrument.lower() == "all":
			self.fermi_config["selection"]["tmin"] = tmin_met
			self.fermi_config["selection"]["tmax"] = tmax_met

		if instrument.lower() == "veritas" or instrument.lower() == "all":
			self.veritas_config["selection"]["tmin"] = tmin_mjd
			self.veritas_config["selection"]["tmax"] = tmax_mjd

		self.set_config(self.config, self._filename)
		self.print_info(self._filename)


	@staticmethod
	def get_config(config_file="config.yaml"):
		"""
	    Read a config file.

	    Args:
	        config_file (str): Fermi config filename (yaml)
	        	Default: config.yaml
	    """
		return yaml.load(open(config_file), Loader=yaml.FullLoader)

	@classmethod
	def print_config(self, config_file="config.yaml"):
		"""
	    print a config file.

	    Args:
	    	config_file (str): Fermi config filename (yaml)
				Default: config.yaml
	    """
		self.config = self.get_config(config_file)
		self.fermi_config = self.config.get("fermi")
		self.veritas_config = self.config.get("veritas")

		if not(hasattr(self, "_logging")):
			self._logging = logger()
		self._logging.info("\n"+yaml.dump(self.config, sort_keys=False, default_flow_style=False))

	@classmethod
	def print_info(self, config_file="config.yaml"):
		self.config = self.get_config(config_file)
		self.fermi_config = self.config.pop("fermi")
		self.veritas_config = self.config.pop("veritas")
		self._logging = logger()
		self._logging.info("-"*20+" Info "+"-"*20)
		self._logging.info("target: {}".format(self.veritas_config["selection"]["target"]))
		self._logging.info("localization:")
		self._logging.info("\t(ra, dec) : ({}, {})".format(self.veritas_config["selection"]["ra"],
		                                    self.veritas_config["selection"]["dec"]))
		self._logging.info("\t(glat, glon) : ({}, {})".format(self.veritas_config["selection"]["glat"],
		                                    self.veritas_config["selection"]["glon"]))
		self._logging.info("time interval:")
		self._logging.info("\tveritas : {} - {}".format(utils.MJD2UTC(self.veritas_config["selection"]["tmin"]),
		                                    utils.MJD2UTC(self.veritas_config["selection"]["tmax"])))
		self._logging.info("\tfermi : {} - {}".format(utils.MET2UTC(self.fermi_config["selection"]["tmin"]),
		                                  utils.MET2UTC(self.fermi_config["selection"]["tmax"])))
		self._logging.info("-"*45)

	@staticmethod
	def set_config(info, config_file="config.yaml"):
		"""
	    Write inputs into a config file.

	    Args:
	    	info (dict): overwrite the input info into a config file
	        config_file (str): Fermi config filename (yaml)
	        	Default: config.yaml
	    """
		with open(config_file, "w") as f:
			yaml.dump(info, f)


	@classmethod
	def update_config(self, info, instrument, config_file="config.yaml"):
		"""
	    Update a config file.

	    Args:
	    	info (dict): update info in a config file
	    	instrument (str): either fermi or veritas
	        config_file (str): Fermi config filename (yaml)
				Default: config.yaml
	    """
		pre_info = self.get_config(config_file)

		info = self._filter(pre_info[instrument], info)

		pre_info[instrument] = self._update(pre_info[instrument], info)

		self.set_config(pre_info, config_file)

	@staticmethod
	def _filter(pre_info, info):
		if len(info) != 0:
			for key in list(info.keys()):
				for subkey in list(info[key].keys()):
					if (pre_info[key][subkey] == info[key][subkey]) or (info[key][subkey]==None):
						info[key].pop(subkey)
		return info

	@staticmethod
	def _update(pre_info, info):
		if len(info) != 0:
			for key in info.keys():
				for subkey in info[key].keys():
					pre_info[key][subkey] = info[key][subkey]
					if (key == "selection") and (subkey=="ra"):
						if info['selection']['ra'] != None and info['selection']['dec'] != None:
							if 'binning' in pre_info.keys():
								pre_info['binning']['coordsys'] = 'CEL'
							glon, glat = utils.CEL2GAL(info['selection']['ra'], info['selection']['dec'])
							pre_info['selection']['glon'], pre_info['selection']['glat'] = float(glon), float(glat)

					if (key == "selection") and (subkey=="glon"):
						if info['selection']['glon'] != None and info['selection']['glat'] != None:
							if 'binning' in pre_info.keys():
								pre_info['binning']['coordsys'] = 'CEL'
							ra, dec = utils.GAL2CEL(info['selection']['glon'], info['selection']['glat'])
							pre_info['selection']['ra'], pre_info['selection']['dec'] = float(ra), float(dec)

		return pre_info

	@staticmethod
	def _empty4fermi(outdir = "./fermi/", datadir = "./fermi/", gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt"):
		if not(os.path.isdir(outdir)):
			os.system(f"mkdir {outdir}")
		if not(os.path.isdir(datadir)):
			os.system(f"mkdir {datadir}")
		if not(os.path.isdir(f"{outdir}/log")):
			os.system(f"mkdir {outdir}/log")
			os.system(f": > {outdir}/log/fermipy.log")

		info = {
 				'data': {
 					'evfile': f"{datadir}/EV00.lst",
 					'scfile': f"{datadir}/SC00.fits",
 					'ltcube': None
 					},
 				'binning': {
 					'roiwidth': 12,
  					'binsz': 0.08,
  					'binsperdec': 8,
  					'coordsys': None,
  					'projtype': 'WCS',
  					},
 				'selection': {
 					'emin': 100,
					'emax': 300000,
					'tmin': None,
					'tmax': None,
					'zmax': 90,
					'evclass': 128,
					'evtype': 3,
					'glon': None,
					'glat': None,
					'ra': None,
					'dec': None,
					'target': None
					},
				'gtlike': {
					'edisp': True,
					'irfs': 'P8R3_SOURCE_V3',
					'edisp_disable': ['isodiff', 'galdiff']
					},
				'model': {
					'src_roiwidth': 15,
					'galdiff': f'$FERMI_DIFFUSE_DIR/{gald}',
					'isodiff': f'$FERMI_DIFFUSE_DIR/{iso}',
					'catalogs': SCRIPT_DIR+'/refdata/gll_psc_v22.fit'
					},
				'fileio': {
					'outdir' : outdir,
   					'logfile' : f"{outdir}/log/fermipy.log",
					'usescratch': False
					},
				}
		return info

	@staticmethod
	def _empty4veritas(outdir = "./veritas/", datadir="./veritas/"):
		if not(os.path.isdir(outdir)):
			os.system(f"mkdir {outdir}")

		info = {
			'background':
			{
				'file': SCRIPT_DIR+"/refdata/Hipparcos_MAG8_1997.dat",
				'distance': 1.75,
				'magnitude': 7,
				'simbad': True,
			},
			'data': {
				'anasum': datadir,
			},
			'fileio':{
				'outdir': outdir,
			},
			'cuts':{
				'th2cut': 0.008,
				'eff_cut': 0,
				'bias_cut': 0,
			},
			'selection':
			{
				'target': None,
				'ra': None,
				'dec': None,
				'tmin' : None,
				'tmax' : None,
				'emin': 0.1,
				'emax': 10,
				'nbin': 6,
				'format': "mjd",
				'max_region_number': 6,
				'radius': 2.0,
				'exc_on_region_radius': 0.7,
				'exc_radius': 0.25,
				},

			}
		return info
