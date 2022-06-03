from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=20)

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
    if verbosity == 1:
        level = 20
    elif verbosity == 2:
        level = 10
    elif verbosity == 0:
        level = 40
    elif verbosity == -1:
        level = 50
    else:
        level = 20
    logging.getLogger().setLevel(level)
    return logging

def METtoUTC(met):
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

def UTCtoMET(utc):
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

def MJDtoUTC(mjd):
	"""
    Convert MJD (Modified Julian Day) to UTC.

    Args:
        mjd (astorpy.time): MJD time
    
    Return:
        astropy.time: UTC
    """
	refMJD = Time(mjd, format='mjd')
	return refMJD.isot

def CELtoGAL(ra, dec):
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

def GALtoCEL(l, b):
	"""
    Convert MJD (Modified Julian Day) to UTC.

    Args:
        mjd (astorpy.time): MJD time
    
    Return:
        astropy.time: UTC
    """
	c = SkyCoord(l=float(l)*u.degree, b=float(b)*u.degree, frame='galactic')
	return c.icrs.ra.deg, c.icrs.dec.deg 
