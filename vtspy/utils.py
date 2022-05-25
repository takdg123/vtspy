from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord

from astropy.io import fits


def METtoUTC(met):
    refMET = Time('2001-01-01', format='isot', scale='utc')
    dt = TimeDelta(met, format='sec')
    return (refMET+dt).iso

def UTCtoMET(utc):
    refMET = Time('2001-01-01', format='isot', scale='utc')
    currentTime = Time(utc, format='isot', scale='utc')
    return float((currentTime-refMET).sec)

def METnow():
	refMET = Time('2001-01-01', format='isot', scale='utc')
	currentTime = Time.now()
	return float((currentTime-refMET).sec)

def MJDtoUTC(mjd):
	refMJD = Time(mjd, format='mjd')
	return refMJD.isot

def CELtoGAL(ra, dec):
	c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
	return c.galactic.l.deg, c.galactic.b.deg

def GALtoCEL(l, b):
	c = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
	return c.icrs.ra.deg, c.icrs.dec.deg 
