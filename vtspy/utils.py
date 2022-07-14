import os
import numpy as np
import copy
import glob
from pathlib import Path

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import logging
logging.basicConfig(format=('%(asctime)s %(levelname)-8s: %(message)s'), datefmt='%Y-%m-%d %H:%M:%S', level=20)

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
        float: MET
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
        float: MJD
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

def convertROOT2fits(files, eff, **kwargs):
    from pyV2DL3.genHDUList import genHDUlist, loadROOTFiles
    from pyV2DL3 import generateObsHduIndex

    if type(files) == str:
        files = [files]
    else:
        files = glob.glob(files + "/*anasum.root")

    full_enclosure = kwargs.pop("full_enclosure", True)
    point_like = kwargs.pop("point_like", True)
    instrument_epoch = kwargs.pop("instrument_epoch", None)
    save_multiplicity = kwargs.pop("save_multiplicity", False)
    filename_to_obsid = kwargs.pop("filename_to_obsid", True)
    evt_filter = kwargs.pop("evt_filter", None)

    if evt_filter is not None:
        evt_filter = Path(evt_filter)

    force_extrapolation = kwargs.get("force_extrapolation", False)
    fuzzy_boundary = kwargs.get("fuzzy_boundary", 0.0)

    if not(full_enclosure) and not(point_like):
        point_like = True
        full_enclosure = False

    irfs_to_store = {"full-enclosure": full_enclosure, "point-like": point_like}

    for file in files:
        datasource = loadROOTFiles(Path(file), Path(eff), "ED")
        datasource.set_irfs_to_store(irfs_to_store)

        datasource.fill_data(
            evt_filter = evt_filter, 
            use_click = False, 
            force_extrapolation = force_extrapolation, 
            fuzzy_boundary = fuzzy_boundary, 
            **kwargs)
    
        hdulist = genHDUlist(
            datasource,
            save_multiplicity = save_multiplicity,
            instrument_epoch = instrument_epoch,
        )

        fname_base = os.path.basename(file)
        obs_id = int(fname_base.split(".")[0])

        if filename_to_obsid:
            hdulist[1].header["OBS_ID"] = obs_id

        output = kwargs.get("output", file.replace(".root", ".fits"))
        if ".fits" not in output:
            output +=".fits"

        hdulist.writeto(output, overwrite=True)

    datadir = str(Path(file).absolute().parent)
    filelist = glob.glob(f"{datadir}/*anasum.fit*")

    generateObsHduIndex.create_obs_hdu_index_file(filelist, index_file_dir=datadir)

def generatePSF(config, **kwargs):

    emin = kwargs.pop("emin", config['selection']['emin'])
    emax = kwargs.pop("emax", config['selection']['emax'])
    binsperdec = config['binning']['binsperdec']
    enumbins = kwargs.pop("enumbins", int((np.log10(emax)-np.log10(emin))*binsperdec))
    ntheta = int(30/config['binning']['binsz'])
    thetamax = config['binning']['binsz']*ntheta
    
    from GtApp import GtApp
    gtpsf = GtApp('gtpsf')
    workdir = config['fileio']['workdir']
    gtpsf["expcube"] = '{}/ltcube_00.fits'.format(workdir)
    gtpsf["outfile"] = kwargs.pop("outfile", '{}/gtpsf_00.fits'.format(workdir))
    gtpsf["irfs"] = config['gtlike']['irfs']
    gtpsf['evtype'] = config['selection']['evtype']
    gtpsf['ra'] = config['selection']['ra']
    gtpsf['dec'] = config['selection']['dec']
    gtpsf['emin'] = emin
    gtpsf['emax'] = emax
    gtpsf['thetamax'] = thetamax
    gtpsf['ntheta'] = ntheta
    gtpsf['nenergies'] = enumbins
    gtpsf['chatter'] = 0
    gtpsf.run()

def generateRSP(config):
    from gt_apps import rspgen
    workdir = config['fileio']['workdir']
    emin = config['selection']['emin']
    emax = config['selection']['emax']
    binsperdec = config['binning']['binsperdec']
    enumbins = int((np.log10(emax)-np.log10(emin))*binsperdec)
    
    rspgen['respalg'] = 'PS'
    rspgen['specfile'] = '{}/gtpha_00.pha'.format(workdir)
    rspgen['scfile'] = config['data']['scfile']
    rspgen['outfile'] = '{}/gtrsp_00.rsp'.format(workdir)
    rspgen['thetacut'] = config['selection']['zmax']
    rspgen['irfs'] = config['gtlike']['irfs']
    rspgen['emin'] = config['selection']['emin']
    rspgen['emax'] = config['selection']['emax']
    rspgen['ebinalg'] = "LOG"
    rspgen['enumbins'] = enumbins
    rspgen['chatter'] = 0
    rspgen.run() 

def generatePHA(config):
    from gt_apps import evtbin
    workdir = config['fileio']['workdir']
    emin = config['selection']['emin']
    emax = config['selection']['emax']
    binsperdec = config['binning']['binsperdec']
    enumbins = int((np.log10(emax)-np.log10(emin))*binsperdec)
    
    evtbin['evfile'] = '{}/ft1_00.fits'.format(workdir)
    evtbin['scfile'] = config['data']['scfile']
    evtbin['outfile'] = '{}/gtpha_00.pha'.format(workdir)
    evtbin['algorithm'] = 'PHA1'
    evtbin['ebinalg'] = 'LOG'
    evtbin['emin'] = config['selection']['emin']
    evtbin['emax'] = config['selection']['emax']
    evtbin['enumbins'] = enumbins
    evtbin['coordsys'] = 'CEL'
    evtbin['xref'] = config['selection']['ra']
    evtbin['yref'] = config['selection']['dec']
    evtbin['chatter'] = 0
    evtbin.run()
