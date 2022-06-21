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
	    info (dict, optional): manual inputs
	    verbosity (int)
	"""

	def __init__(self, files=None, config_file="config.yaml", info = {}, verbosity=1, **kwargs):
		self._logging = logger(verbosity=verbosity)

		path = Path(config_file)
		if path.is_file() and (files is None):
			config = self.get_config(config_file)
			self.fermi_config = config.pop("fermi")
			self.veritas_config = config.pop("veritas")
			self._logging.info(f'a configuration file ({config_file}) is loaded.') 
		else:
			self.init(files=files, config_file = config_file, info=info, **kwargs)
			self._logging.info(f'a configuration file ({config_file}) is created.') 


	def init(self, files, config_file="config.yaml", info = {}, verbosity=1, **kwargs):

		"""
	    Initiate to generate a config file

	    Args:
	        file (str): name of the input file (.root or .fits)
	        directory (str): name of a directory containing root or fits files
	        config_file (str): Fermi config filename (yaml)
	        	Default: config.yaml
	        info (dict, optional): manual inputs
	        verbosity (int)
		"""

		gald = kwargs.pop("gald", "gll_iem_v07.fits")
		iso = kwargs.pop("iso", "iso_P8R3_SOURCE_V3_v1.txt")
		
		if files is not None:
			filelist = glob.glob(files+"*")
		
		for file in filelist:
			if ".gz" in file:
				filelist.remove(file)

		self.fermi_config = self._empty4fermi(gald=gald, iso=iso)	
		self.veritas_config = self._empty4veritas()	
		
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
		
		info = {"fermi": self.fermi_config, "veritas": self.veritas_config}
		
		self.set_config(info, config_file)

		self.config = info

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

		if not(hasattr(self, "_logging")):
			self._logging = logger()
		self._logging.info("\n"+yaml.dump(self.config, sort_keys=False, default_flow_style=False))

	
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
	def _empty4fermi(gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt"):
		if not(os.path.isdir("./fermi")):
			os.system("mkdir fermi")
		if not(os.path.isdir("./fermi/log")):
			os.system("mkdir ./fermi/log")
			os.system(": > ./fermi/log/fermipy.log")

		info = {
 				'data': {
 					'evfile': "./fermi/EV00.lst",
 					'scfile': "./fermi/SC00.fits",
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
					'zmax': 105,
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
					'src_roiwidth': 12,
					'galdiff': f'$FERMI_DIFFUSE_DIR/{gald}',
					'isodiff': f'$FERMI_DIFFUSE_DIR/{iso}',
					'catalogs': SCRIPT_DIR+'/refdata/gll_psc_v22.fit'
					},
				'fileio': {
					'outdir' : "./fermi/",
   					'logfile' : "./fermi/log/fermipy.log",
					'usescratch': False
					},
				}
		return info

	@staticmethod
	def _empty4veritas():
		if not(os.path.isdir("./veritas")):
			os.system("mkdir veritas")

		info = {
			'background':
			{
				'file': SCRIPT_DIR+"/refdata/Hipparcos_MAG8_1997.dat",
				'distance': 1.75,
				'magnitude': 7,
			},
			'fileio':{
				'outdir': "./veritas/",
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
				'emax': 2.0,
				'format': "mjd",
				'max_region_number': 6,
				'radius': 2.0,
				'exc_on_region_radius': 0.7,
				'exc_radius': 0.25,
				},

			}
		return info
		
