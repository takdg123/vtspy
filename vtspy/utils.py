import os
import numpy as np
import copy

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import logging
logging.basicConfig(format=('%(asctime)s %(levelname)-8s: %(message)s'), datefmt='%Y-%m-%d %H:%M:%S', level=20)

from pathlib import Path
SCRIPT_DIR = str(Path(__file__).parent.absolute())

MeV2Erg = 1.60218e-6
TeV2Erg = 1.60218

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

def MET2UTC(met, return_astropy=False):
    """
    Convert Fermi MET (Mission Elapsed Time) to UTC (Coordinated 
    Universal Time).

    Args:
        met (float): MET time in seconds
        return_astropy (bool): return astropy.time
    
    Return:
        str, astropy.time (optional): UTC time
    """

    refMET = Time('2001-01-01', format='isot', scale='utc')

    dt = TimeDelta(met, format='sec')

    if return_astropy:
        return (refMET+dt).iso, refMET+dt
    else:
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

def MJD2UTC(mjd, return_astropy=False):
    """
    Convert MJD (Modified Julian Day) to UTC.

    Args:
        mjd (astorpy.time): MJD time
        return_astropy (bool): return astropy.time

    Return:
        str, astropy.time (optional): UTC time
    """

    refMJD = Time(mjd, format='mjd')
    if return_astropy:
        return refMJD.isot, refMJD
    else:
        return refMJD.isot

def UTC2MJD(utc):
    """
    Convert UTC to MJD.

    Args:
        mjd (astorpy.time): MJD time
    
    Return:
        astropy.time: UTC
    """
    refUTC= Time(utc, format='isot', scale='utc')
    return float(refUTC.mjd)

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

def bright_source_list(source, save_npy=True):

    if ".npy" in source:
        bright_sources = np.load(source)
    else:
        bright_sources = []
        with open(source) as f: 
            for line in f.readlines()[27:]:
                info = line.split()
                if len(info) == 5:
                    ra = info[0]
                    dec = info[1]
                    bright = info[3]
                    brightbv = info[4]

                    bright_sources.append([float(ra), float(dec), float(bright), float(brightbv)])
            if save_npy:
                source = source.replace(".dat", ".npy")
                np.save(source, bright_sources)
    return np.asarray(bright_sources)

def time_filter(obs, time, time_format=None):
    """
    Filter the observations by time
    
    Args:
        obs (gammapy.observations): the list of observations
        time (list): the list of times
        time_format (str, optional): passed to astropy.Time
    """

    obs = copy.deepcopy(obs)

    if type(time[0]) == Time:
        time_interval = time
    else:
        tmin, tmax = time[0], time[1]
        if time_format == "MET":
            tmin = MET2UTC(tmin)
            tmax = MET2UTC(tmax)
            time_interval= Time([str(tmin),str(tmax)])
        else:
            time_interval= Time([str(tmin),str(tmax)], format=time_format, scale="utc")
    
    newobs = obs.select_time(time_interval)
    
    return newobs, newobs.ids

def LiMaSiginficance(N_on, N_off, alpha, type=1):
    """
    Calculate the Li&Ma significance
    
    Args:
        N_on (int): the number of events from the ON region
        N_off (int): the number of events from the OFF region
        alpha (float): the relative exposure time
        type (int): method for calculating signficance 
            1: Li&MA, 2: chisq
            Default: 1
    Return
        float: Li&Ma significance

    """

    if type == 1:
        temp = N_on*np.log((1.+alpha)/alpha*(N_on/(N_on+N_off)))+N_off*np.log((1+alpha)*(N_off/(N_on+N_off)))
    
        if np.size(temp) != 1:
            for i, t in enumerate(temp):
                if t > 0:
                    temp[i] = np.sqrt(t)
                else:
                    temp[i] = np.nan
        else:
            if temp >0:
                temp = np.sqrt(temp)
            else:
                temp = np.nan

        significance = np.sign(N_on-alpha*N_off)*np.sqrt(2.)*temp
    else:
        significance = (N_on-alpha*N_off)/np.sqrt(alpha*(N_on+N_off))
    return significance


def define_time_intervals(tmin, tmax, binsz=None, nbins=None):
    """
    Define time intervals by either a bin size or the number of bins
    
    Args:
        tmin (float): minimum time
        tmax (float): maximum time
        binsz (astropy.Quantity, optional): bin size with astropy.units
        nbins (int, optional): the number of bins
    
    Return:
        list: time interval (astropy.Time)
    """
    if binsz is not None:
        _, tmin = MJD2UTC(tmin, return_astropy=True)
        _, tmax = MJD2UTC(tmax, return_astropy=True)
        nbins = int((tmax-tmin)*u.second/binsz.to(u.second))
        nbins +=1
        times = tmin + np.arange(nbins) * binsz
    elif nbins is not None:
        _, tmin = MJD2UTC(tmin, return_astropy=True)
        _, tmax = MJD2UTC(tmax, return_astropy=True)
        binsz = ((tmax-tmin)/nbins).to(u.second)
        nbins +=1
        times = tmin + np.arange(nbins) * binsz

    if len(times) == 1:
        time_intervals = [Time([tmin, tmax])]
    else:
        time_intervals = [Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])]

    return time_intervals