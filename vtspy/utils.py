import os
import numpy as np

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import logging
logging.basicConfig(format=('%(asctime)s %(levelname)-8s: %(message)s'), datefmt='%Y-%m-%d %H:%M:%S', level=20)

from pathlib import Path
SCRIPT_DIR = str(Path(__file__).parent.absolute())



def logger(verbosity = 1):
    """
    Set a log level:

    * 1: info, warning, error
    * 2: debug, info, warning, error
    * 0: warning, error
    * -1: error

    Args:
        verbosity (int)
            Default: 1
    
    Return:
        astorpy.time: UTC time
    """
    levels_dict = {2: 10,
                   1: 20,
                   0: 30,
                   -1: 40,
                   -2: 50}
                   
    level = levels_dict[verbosity]
    logging.getLogger().setLevel(level)
    return logging

def MET2UTC(met):
    """
    Convert Fermi MET (Mission Elapsed Time) to UTC (Coordinated 
    Universal Time).

    Args:
        met (float): MET time in seconds
    
    Return:
        astorpy.time: UTC time
    """

    refMET = Time('2001-01-01', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    return (refMET+dt).iso

def UTC2MET(utc):
    """
    Convert UTC to Fermi MET (mission elapsed time).

    Args:
        utc (astorpy.time): UTC time
    
    Return:
        float: Fermi MET
    """
    refMET = Time('2001-01-01', format='isot', scale='utc')
    currentTime = Time(utc, format='isot', scale='utc')
    return float((currentTime-refMET).sec)

def METnow():
	"""
    Return the current MET.

    Return:
        float: Fermi MET
    """
	refMET = Time('2001-01-01', format='isot', scale='utc')
	currentTime = Time.now()
	return float((currentTime-refMET).sec)

def MJD2UTC(mjd):
	"""
    Convert MJD (Modified Julian Day) to UTC.

    Args:
        mjd (astorpy.time): MJD time
    
    Return:
        astropy.time: UTC
    """
	refMJD = Time(mjd, format='mjd')
	return refMJD.isot

def CEL2GAL(ra, dec):
	"""
    Convert CEL (celestial) coordinates to GAL (galactic) coordinates.

    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees
    
    Return:
        deg, deg
    """
	c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs')
	return c.galactic.l.deg, c.galactic.b.deg

def GAL2CEL(l, b):
	"""
    Convert MJD (Modified Julian Day) to UTC.

    Args:
        mjd (astorpy.time): MJD time
    
    Return:
        astropy.time: UTC
    """
	c = SkyCoord(l=float(l)*u.degree, b=float(b)*u.degree, frame='galactic')
	return c.icrs.ra.deg, c.icrs.dec.deg 

def bright_source_list(source = "Hipparcos_MAG8_1997", save_npy=True):

    if os.path.exists(SCRIPT_DIR+"/refdata/"+source+".npy"):
        bright_sources = np.load(SCRIPT_DIR+"/refdata/"+source+".npy")
    else:
        bright_sources = []
        with open(SCRIPT_DIR+"/refdata/"+source+".dat") as f: 
            for line in f.readlines()[27:]:
                info = line.split()
                if len(info) == 5:
                    ra = info[0]
                    dec = info[1]
                    bright = info[3]
                    brightbv = info[4]

                    bright_sources.append([float(ra), float(dec), float(bright), float(brightbv)])
            if save_npy:
                np.save(SCRIPT_DIR+"/refdata/"+source, bright_sources)
    return np.asarray(bright_sources)
