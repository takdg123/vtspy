
import os
from astropy.time import Time
import yaml
from . import utils
import uproot
import glob
from astropy.io import fits

SCRIPT_DIR = os.environ.get('vtspy')

class config:
	def __init__(self, file=None, config_file="config_fermi.yaml", info = {}):
		try:
			self.file = self.getConfig(config_file)
		except:
			self.init(file=file, config_file = config_file, info=info)
		return			

	@classmethod
	def init(self, file=None, config_file="config_fermi.yaml", info = {}):
		
		if file !=None:
			filelist = glob.glob(file)
		else:
			filelist = []

		data = self.__emptyfile__()	

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
							print("[Error] RA values in input files are different.")
							
					if dec == None:
						dec = float(tRun['TargetDecJ2000'].arrays(library="np")['TargetDecJ2000'][0])
					else:
						temp = float(tRun['TargetDecJ2000'].arrays(library="np")['TargetDecJ2000'][0])
						if temp != dec:
							print("[Error] DEC values in input files are different.")		
					
					tmin = utils.MJDtoUTC(tRun['MJDOn'].arrays(library="np")['MJDOn'][0])
					tmin = utils.UTCtoMET(tmin[:10])
					tmax = utils.MJDtoUTC(tRun['MJDOff'].arrays(library="np")['MJDOff'][0])
					tmax = utils.UTCtoMET(tmax[:10])+60*60*24

					if info['selection']['tmin'] < tmin:
						tmin = info['selection']['tmin']

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
							print("[Error] RA values in input files are different.")
					
					if dec == None:
						dec = header['DEC_OBJ']
					else:
						temp = header['DEC_OBJ']
						if temp != dec:
							print("[Error] DEC values in input files are different.")		
					
					tmin = header['DATE-OBS']
					tmin = utils.UTCtoMET(tmin[:10])
					tmax = header['DATE-END']
					tmax = utils.UTCtoMET(tmax[:10])+60*60*24
					
					if info['selection']['tmin'] != None:
						if info['selection']['tmin'] < tmin:
							tmin = info['selection']['tmin']
					if info['selection']['tmax'] != None:
						if info['selection']['tmax'] > tmax:
							tmax = info['selection']['tmax']	

					info = {**info, 'selection':{'ra': ra, 'dec': dec, 'tmin': tmin, 'tmax':tmax}}

		
		info = self.__filter__(data, info)

		file = self.__update__(data, info)

		self.setConfig(data, config_file)

		
	@staticmethod
	def getConfig(config_file="config_fermi.yaml"):
		return yaml.load(open(config_file), Loader=yaml.FullLoader)
	
	@staticmethod
	def setConfig(data, config_file="config_fermi.yaml"):		
		with open(config_file, "w") as f:
			yaml.dump(data, f)

	
	@classmethod
	def updateConfig(self, info, config_file="config_fermi.yaml"):
		data = self.getConfig(config_file)
		
		info = self.__filter__(data, info)
		
		data = self.__update__(data, info)
		
		self.setConfig(data, config_file)

	@classmethod
	def printConfig(self, config_file="config_fermi.yaml"):
		data = self.getConfig(config_file)
		print(yaml.dump(data, sort_keys=False, default_flow_style=False))


	def __filter__(data, info):
		if len(info) != 0:
			for key in list(info.keys()):
				for subkey in list(info[key].keys()):
					if (data[key][subkey] == info[key][subkey]) or (info[key][subkey]==None):
						info[key].pop(subkey)
		return info

	def __update__(data, info):
		if len(info) != 0:
			for key in info.keys():
				for subkey in info[key].keys():
					data[key][subkey] = info[key][subkey]
					
					if (key == "selection") and (subkey=="ra"):
						if info['selection']['ra'] != None and info['selection']['dec'] != None:
							data['binning']['coordsys'] = 'CEL'
							data['selection']['glon'], data['selection']['glat'] = utils.CELtoGAL(info['selection']['ra'], info['selection']['dec'])

					if (key == "selection") and (subkey=="glon"):
						if info['selection']['glon'] != None and info['selection']['glat'] != None:
							data['binning']['coordsys'] = 'CEL'
							data['selection']['ra'], data['selection']['dec'] = utils.GALtoCEL(info['selection']['glon'], info['selection']['glat'])

		return data

	def __emptyfile__():
		if not(os.path.isdir("./fermi")):
			os.system("mkdir fermi")
		if not(os.path.isdir("./log")):
			os.system("mkdir log")
			os.system(": > ./log/fermipy.log".format(info[key][subkey]))

		data = {
 				'data': {
 					'evfile': "./fermi/SC00.fits",
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
					'galdiff': '$FERMI_DIFFUSE_DIR/gll_iem_v07.fits',
					'isodiff': '$FERMI_DIFFUSE_DIR/iso_P8R3_SOURCE_V3_v1.txt',
					'catalogs': SCRIPT_DIR+'/vtspy/refdata/gll_psc_v22.fit'
					},
				'fileio': {
					'outdir' : "./fermi/",
   					'logfile' : "./log/fermipy.log",
					'usescratch': False
					}
				}
		return data
