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

	def __init__(self, files=None, config_file="config.yaml", gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt", info = {}, verbosity=1):
		self._logging = logger(verbosity=verbosity)

		path = Path(config_file)
		if path.is_file() and (files is None):
			self.config = self.get_config(config_file)
			self._logging.info(f'a configuration file ({config_file}) is loaded.') 
		else:
			self.init(files=files, config_file = config_file, info=info, gald=gald, iso=iso)
			self._logging.info(f'a configuration file ({config_file}) is created.') 


	@classmethod
	def init(self, files, config_file="config.yaml", gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt", info = {}, verbosity=1):

		"""
	    Initiate to generate a Fermipy config file

	    Args:
	        file (str): name of the input file (.root or .fits)
	        directory (str): name of a directory containing root or fits files
	        config_file (str): Fermi config filename (yaml)
	        	Default: config.yaml
	        info (dict, optional): manual inputs
	        verbosity (int)
		"""
		
		if files is not None:
			filelist = glob.glob(files+"*")
		
		for file in filelist:
			if ".gz" in file:
				filelist.remove(file)

		pre_info = self.__emptyfile__(gald=gald, iso=iso)	

		info = {**info, 
			'selection':{
				'ra': None, 
				'dec': None, 
				'tmin': None, 
				'tmax':None},
			'vts_setup':{
				'tmin': None,
				'tmax': None,
			}}
		
		ra = info['selection']['ra']
		dec = info['selection']['dec']
		
		for file in filelist:

			if file!=None:

				if '.root' in file:

					anasum = uproot.open(file)

					tRun = anasum['total_1']['stereo']['tRunSummary']
					
					if info['selection']['ra'] == None:
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
					
				elif '.fits' in file:

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
				
				if info['selection']['tmin'] is not None:
					if info['selection']['tmin'] < tmin:
						tmin = info['selection']['tmin']
					else:
						info['selection']['tmin'] = tmin
				
				if info['selection']['tmax'] is not None:
					if info['selection']['tmax'] > tmax:
						tmax = info['selection']['tmax']
					else:
						info['selection']['tmax'] = tmax
						
				if info['vts_setup']['tmin'] is not None:
					if info['vts_setup']['tmin'] < tmin_mjd:
						tmin_mjd = info['vts_setup']['tmin']
					else:
						info['vts_setup']['tmin'] = tmin_mjd
				if info['vts_setup']['tmax'] is not None:
					if info['vts_setup']['tmax'] > tmax_mjd:
						tmax_mjd = info['vts_setup']['tmax']
					else:
						info['vts_setup']['tmax'] = tmax_mjd

				
				info = {**info, 
					'selection':{
						'ra': ra, 
						'dec': dec, 
						'tmin': tmin, 
						'tmax': tmax, 
						'target': target},
					'vts_setup':{
						'tmin': tmin_mjd, 
						'tmax': tmax_mjd, 
					}}


		
		info = self._filter(pre_info, info)

		info = self._update(pre_info, info)

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
	def update_config(self, info, config_file="config.yaml"):
		"""
	    Update a config file.

	    Args:
	    	info (dict): update info in a config file
	        config_file (str): Fermi config filename (yaml)
				Default: config.yaml
	    """
		pre_info = self.get_config(config_file)
		
		info = self._filter(pre_info, info)
		
		pre_info = self._update(pre_info, info)
		
		self.set_config(pre_info, config_file)

	def print_config(self, config_file="config.yaml"):
		"""
	    print a config file.

	    Args:
	    	config_file (str): Fermi config filename (yaml)
				Default: config.yaml
	    """
		if not(hasattr(self, "config")):
			self.config = self.get_config(config_file)
		self._logging.info("\n"+yaml.dump(self.config, sort_keys=False, default_flow_style=False))

	def _filter(pre_info, info):
		if len(info) != 0:
			for key in list(info.keys()):
				for subkey in list(info[key].keys()):
					if (pre_info[key][subkey] == info[key][subkey]) or (info[key][subkey]==None):
						info[key].pop(subkey)
		return info

	def _update(pre_info, info):
		if len(info) != 0:
			for key in info.keys():
				for subkey in info[key].keys():
					pre_info[key][subkey] = info[key][subkey]
					if (key == "selection") and (subkey=="ra"):
						if info['selection']['ra'] != None and info['selection']['dec'] != None:
							pre_info['binning']['coordsys'] = 'CEL'
							glon, glat = utils.CEL2GAL(info['selection']['ra'], info['selection']['dec'])
							pre_info['selection']['glon'], pre_info['selection']['glat'] = float(glon), float(glat)

					if (key == "selection") and (subkey=="glon"):
						if info['selection']['glon'] != None and info['selection']['glat'] != None:
							pre_info['binning']['coordsys'] = 'CEL'
							ra, dec = utils.GAL2CEL(info['selection']['glon'], info['selection']['glat'])
							pre_info['selection']['ra'], pre_info['selection']['dec'] = float(ra), float(dec)

		return pre_info

	def __emptyfile__(gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt"):
		if not(os.path.isdir("./fermi")):
			os.system("mkdir fermi")
		if not(os.path.isdir("./veritas")):
			os.system("mkdir veritas")
		if not(os.path.isdir("./log")):
			os.system("mkdir log")
			os.system(": > ./log/fermipy.log")

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
   					'logfile' : "./log/fermipy.log",
					'usescratch': False
					},
				'vts_setup':{
					'tmin' : None,
					'tmax' : None,
					'format': "mjd",
					'max_region_number': 6,
					'radius': 2.0,
					'th2cut': 0.008,
					'exc_on_region_radius': 0.7,
					'exc_radius': 0.25,
					'emin': 0.1,
					'emax': 2.0,
					'eff_cut': 0,
					'bias_cut': 0,
					'outdir':  "./veritas/",
					'bright_srcs': ['Hipparcos_MAG8_1997', 1.75, 7],
					},
				}
		return info
