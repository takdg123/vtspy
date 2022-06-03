import os
import glob

import yaml

import uproot

from astropy.io import fits

from . import SCRIPT_DIR
from . import utils
from .utils import logger

from pathlib import Path

class FermipyConfig:
	"""
	This is to generate the configuration file compatible to
	the Fermipy configuration. The input file is the VERITAS
	event file from EventDisplay. The format can be either
	`.root' or `.fits'. 

	Args:
	    files (str): a input file or directory containing 
	    	root or fits files (root or fits)
	    config_file (str): the name of a Fermi config file (.yaml)
	    	Default: config.yaml
	    info (dict, optional): manual inputs
	    verbosity (int)
	"""

	def __init__(self, files=None, config_file="config.yaml", gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt", info = {}, verbosity=1):
		logging = logger(verbosity=verbosity)

		path = Path(config_file)
		if path.is_file() and (files is None):
			self.file = self.getConfig(config_file)
			logging.info(f'a configuration file ({config_file}) is loaded.') 
		else:
			self.init(files=files, config_file = config_file, info=info, gald=gald, iso=iso)
			logging.info(f'a configuration file ({config_file}) is created.') 

	@classmethod
	def init(self, files, config_file="config.yaml", gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt", info = {}, verbosity=1):

		"""
	    Initiate to generate a Fermipy config file

	    Args:
	        file (str): name of the input file (.root or .fits)
	        directory (str): name of a directory containing root or fits files
	        config_file (str): the name of a Fermi config file (.yaml)
	        	Default: config.yaml
	        info (dict, optional): manual inputs
	        verbosity (int)
		"""

		logging = logger(verbosity=verbosity)
		
		if files is not None:
			filelist = glob.glob(files+"*")

		pre_info = self.__emptyfile__(gald=gald, iso=iso)	

		info = {**info, 'selection':{'ra': None, 'dec': None, 'tmin': None, 'tmax':None}}
		
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
							logging.error("[Error] RA values in input files are different.")
							
					if dec == None:
						dec = float(tRun['TargetDecJ2000'].arrays(library="np")['TargetDecJ2000'][0])
					else:
						temp = float(tRun['TargetDecJ2000'].arrays(library="np")['TargetDecJ2000'][0])
						if temp != dec:
							logging.error("[Error] DEC values in input files are different.")		
					
					tmin = utils.MJDtoUTC(tRun['MJDOn'].arrays(library="np")['MJDOn'][0])
					tmin = utils.UTCtoMET(tmin[:10])
					tmax = utils.MJDtoUTC(tRun['MJDOff'].arrays(library="np")['MJDOff'][0])
					tmax = utils.UTCtoMET(tmax[:10])+60*60*24

					
					if info['selection']['tmin'] is not None:
						if info['selection']['tmin'] < tmin:
							tmin = info['selection']['tmin']
					
					if info['selection']['tmax'] is not None:
						if info['selection']['tmax'] > tmax:
							tmax = info['selection']['tmax']
					
					info = {**info, 'selection':{'ra': ra, 'dec': dec, 'tmin': tmin, 'tmax':tmax}}
				elif '.fits' in file:

					header = fits.open(file)[1].header
					
					if ra == None:
						ra = header['RA_OBJ']
					else:
						temp = header['RA_OBJ']
						if temp != ra:
							logging.error("[Error] RA values in input files are different.")
					
					if dec == None:
						dec = header['DEC_OBJ']
					else:
						temp = header['DEC_OBJ']
						if temp != dec:
							logging.error("[Error] DEC values in input files are different.")		
					
					tmin = header['DATE-OBS']
					tmin = utils.UTCtoMET(tmin[:10])
					tmax = header['DATE-END']
					tmax = utils.UTCtoMET(tmax[:10])+60*60*24
					
					target = header['OBJECT']

					if info['selection']['tmin'] is not None:
						if info['selection']['tmin'] < tmin:
							tmin = info['selection']['tmin']
					if info['selection']['tmax'] is not None:
						if info['selection']['tmax'] > tmax:
							tmax = info['selection']['tmax']


					info = {**info, 'selection':{'ra': ra, 'dec': dec, 'tmin': tmin, 'tmax':tmax, 'target': target}}

		
		info = self.__filter__(pre_info, info)

		info = self.__update__(pre_info, info)

		self.setConfig(info, config_file)

		
	@staticmethod
	def getConfig(config_file="config.yaml"):
		"""
	    Read a config file.

	    Args:
	        config_file (str): Fermi config file (.yaml)
	        	Default: config.yaml
	    """
		return yaml.load(open(config_file), Loader=yaml.FullLoader)
	
	@staticmethod
	def setConfig(info, config_file="config.yaml"):		
		"""
	    Write inputs into a config file.

	    Args:
	    	info (dict): overwrite the input info into a config file
	        config_file (str): the name of a Fermipy config file (.yaml)
	        	Default: config.yaml
	    """
		with open(config_file, "w") as f:
			yaml.dump(info, f)

	
	@classmethod
	def updateConfig(self, info, config_file="config.yaml"):
		"""
	    Update a config file.

	    Args:
	    	info (dict): update info in a config file
	        config_file (str): the name of a Fermipy config file (.yaml)
				Default: config.yaml
	    """
		pre_info = self.getConfig(config_file)
		
		info = self.__filter__(pre_info, info)
		
		pre_info = self.__update__(pre_info, info)
		
		self.setConfig(pre_info, config_file)

	@classmethod
	def printConfig(self, config_file="config.yaml"):
		"""
	    print a config file.

	    Args:
	    	config_file (str): the name of a Fermipy config file (.yaml)
				Default: config.yaml
	    """
		pre_info = self.getConfig(config_file)
		print(yaml.dump(pre_info, sort_keys=False, default_flow_style=False))

	def __filter__(pre_info, info):
		if len(info) != 0:
			for key in list(info.keys()):
				for subkey in list(info[key].keys()):
					if (pre_info[key][subkey] == info[key][subkey]) or (info[key][subkey]==None):
						info[key].pop(subkey)
		return info

	def __update__(pre_info, info):
		if len(info) != 0:
			for key in info.keys():
				for subkey in info[key].keys():
					pre_info[key][subkey] = info[key][subkey]
					#print(pre_info['selection']['ra'], pre_info['selection']['dec'])
					if (key == "selection") and (subkey=="ra"):
						if info['selection']['ra'] != None and info['selection']['dec'] != None:
							pre_info['binning']['coordsys'] = 'CEL'
							glon, glat = utils.CELtoGAL(info['selection']['ra'], info['selection']['dec'])
							pre_info['selection']['glon'], pre_info['selection']['glat'] = float(glon), float(glat)

					if (key == "selection") and (subkey=="glon"):
						if info['selection']['glon'] != None and info['selection']['glat'] != None:
							pre_info['binning']['coordsys'] = 'CEL'
							pre_info['selection']['ra'], pre_info['selection']['dec'] = utils.GALtoCEL(info['selection']['glon'], info['selection']['glat'])

		return pre_info

	def __emptyfile__(gald = "gll_iem_v07.fits", iso = "iso_P8R3_SOURCE_V3_v1.txt"):
		if not(os.path.isdir("./fermi")):
			os.system("mkdir fermi")
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
					}
				}
		return info
