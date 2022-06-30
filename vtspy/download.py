import numpy as np
import os, sys
import urllib
import time
import yaml
import html2text
from pathlib import Path

from astropy.io import fits

import re

from .config import JointConfig
from .utils import logger

class DownloadFermiData:
    
    def __init__(self, config_file = "config.yaml", dtype="Photon", verbosity = 1):
        """
        This is to download Fermi-LAT data based on the configuration
        file generated by the 'FermipyConfig' class. 

        Args:
            config_file (str): Fermi config filename (yaml)
                Default: config.yaml 
            dtype (str): either Photon or Extended
                Default: Photon
            verbosity (int)
        """
        self._logging = logger(verbosity=verbosity)

        self.config_file = config_file
        self.config = JointConfig.get_config(self.config_file).pop("fermi")
        path = Path(self.config["data"]["evfile"])
        self.datadir = str(path.parent)
        self.outdir = self.config["fileio"]["outdir"]

        if self.config['selection']['target'] == None:
            self.target = "source"
        else:
            self.target = self.config['selection']['target']

        self.coordsys = self.config['binning']['coordsys']
        if self.coordsys == "GAL":
            self.coord = "Galactic"
            self.loc = [self.config['selection']['glon'], self.config['selection']['glat']]
        elif self.coordsys == "CEL":
            self.coord = "J2000"
            self.loc = [self.config['selection']['ra'], self.config['selection']['dec']]
        
        if self.loc[0] == None or self.loc[1] == None:
            self._logging.error("[Error] Coordinates (e.g., RA & DEC) is not specified.")
            return
        
        self.tmin = self.config['selection']['tmin']
        self.tmax = self.config['selection']['tmax']    
        if self.tmin == None or self.tmax == None:
            self._logging.error("[Error] Time range is not specfied.")
            return

        self.emin = min(100, self.config['selection']['emin'])
        self.emax = max(300000, self.config['selection']['emax'])
        self.dtype = dtype

        success = self._query()

        if success:
            self._download(datadir=self.datadir)


    def _query(self):

        url                         = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi"
        parameters                  = {}
        parameters['coordfield']    = "%s,%s" %(self.loc[0], self.loc[1])
        parameters['coordsystem']   = "%s" %(self.coord)
        parameters['shapefield']    = "%s" %(15)
        parameters['timefield']     = "%s,%s" %(self.tmin,self.tmax)
        parameters['timetype']      = "%s" %("MET")
        parameters['energyfield']   = "%s,%s" %(self.emin,self.emax)
        parameters['photonOrExtendedOrNone'] = "%s" %(self.dtype)
        parameters['destination']   = 'query'
        parameters['spacecraft']    = 'checked'

        self._logging.info("Query parameters:")
        for k,v in parameters.items():
            self._logging.info("%30s = %s" %(k,v))

        postData                    = urllib.parse.urlencode(parameters).encode("utf-8")
        temporaryFileName           = "__temp_query_result.html"
        try:
            os.remove(temporaryFileName)
        except:
            pass
        pass

        urllib.request.urlcleanup()

        urllib.request.urlretrieve(url, temporaryFileName, lambda x,y,z:0, postData)

        with open(temporaryFileName) as htmlFile:
            lines = []
            for line in htmlFile:
                lines.append(line.encode('utf-8'))

            html = "".join(str(lines)).strip()
        
        self._logging.debug("Answer from the LAT data server:")
        
        text = html2text.html2text(html.strip()).split("\n")
        text = list(filter(lambda x:x.find("[") < 0 and  x.find("]") < 0 and x.find("#") < 0 and x.find("* ") < 0 and
                        x.find("+") < 0 and x.find("Skip navigation")<0,text))
        text = list(filter(lambda x:len(x.replace(" ",""))>1,text))
        text = [t for t in text if t[0] != '\\']

        
        for t in text:
            if "occurs after data end MET" in t:
                maxTime = re.findall("occurs after data end MET \(([0-9]+)\)", t)[0]
                self._logging.error("[Error] The current Fermi Data Server does not have data upto the entered 'tmax'.")
                self._logging.error("[Error] 'tmax' value in the config file is changed to the maximum value.")
                self._logging.error("[Error] config['selection']['tmax'] = ", maxTime)
                self._logging.error("[Error] Please try again.")
                self.config['selection']['tmax'] = float(maxTime)
                JointConfig.updateConfig(self.config, self.config_file)
                return False


        text[-3] = text[-3]+" "+text[-2]
        text.remove(text[-2])
        text[-2] = text[-2]+text[-1]
        text.remove(text[-1])

        for t in text: self._logging.debug(t)

        os.remove(temporaryFileName)
        for ln, t in enumerate(text):
            estimatedTimeForTheQuery = re.findall("The estimated time for your query to complete is ([0-9]+) seconds",t)
            if len(estimatedTimeForTheQuery)>0:
                estimatedTimeForTheQuery = estimatedTimeForTheQuery[0]
                break

        for ln, t in enumerate(text):
            address = re.findall("your query may be found ([a-z]+) ",t)
            if len(address)>0:
                line_number = ln
                break

        address = text[line_number] + text[line_number+1]
        self.httpAddress = address.split()[-2][1:-2]
        
        startTime = time.time()
        timeout = 2.*max(5.0,float(estimatedTimeForTheQuery))
        regexpr = re.compile("wget (.*.fits)")

        links = None
        fakeName = "__temp__query__result.html"

        overTime = False

        self._logging.info("The estimated time is about "+str(int(estimatedTimeForTheQuery))+" seconds.")
        while(time.time() <= startTime+timeout):
            remainedTime = int(int(estimatedTimeForTheQuery) - (time.time()-startTime))
            if remainedTime<0 and not(overTime):
                overTime=True
                self._logging.info("The Fermi data is still not ready. Wait for another " + str(int(estimatedTimeForTheQuery)) + " seconds.")
            try:
                (filename, header) = urllib.request.urlretrieve(self.httpAddress,fakeName)
            except:
                urllib.request.urlcleanup()
                continue
            
            with open(fakeName) as f:
                html = " ".join(f.readlines())
                try:
                    status = re.findall("The state of your query is ([0-9]+)",html)[0]
                except:
                    status = '0'
                    pass

                if(status=='2'):
                    links = regexpr.findall(html)
                    if len(links) >= 2:
                        break
                
            os.remove(fakeName)
            urllib.request.urlcleanup()

        try:
            os.remove(fakeName)
        except:
            self._logging.error("[Error] The files (SC and EV files) are not ready to be downloaded. Check the link and then use 'DownloadFermiData.manual_download()' when the data is ready.")
            return
        
        np.save(f"{self.outdir}/fermi_dwn_link", links)

        return True

    def _download(self, datadir="./fermi/"):

        links = np.load(f"{self.outdir}/fermi_dwn_link.npy")

        for lk in links:
            self._logging.info("Downloading... "+lk)
            fileName = lk[-9:-5]

            if "SC" in lk:
                self.config['data']['scfile'] = f"{datadir}/{fileName}.fits"
            elif "EV" in lk:
                with open(f"{datadir}/EV00.lst", "a") as f:
                    f.write(fileName+".fits\n")

            urllib.request.urlretrieve(lk, f"{datadir}/{fileName}.fits")

        self._logging.info("Downloading the Fermi-LAT data has been completed.")
        os.system(f"rm {self.outdir}/fermi_dwn_link.npy")
        JointConfig.update_config(self.config, "fermi", self.config_file)

    def manual_download(self, address=None):
        """
        When an error occurs, one can manually download the data
        
        Args:
            address (str, optional): address to the Fermi-LAT data page.
        """
        if address!=None:
            self.httpAddress = address

        links = None
        fakeName = "__temp__query__result.html"
        regexpr = re.compile("wget (.*.fits)")

        (filename, header) = urllib.request.urlretrieve(self.httpAddress,fakeName)
        
        with open(fakeName) as f:
            html = " ".join(f.readlines())
            links = regexpr.findall(html)
        
        os.remove(fakeName)
        urllib.request.urlcleanup()
        
        np.save(f"{self.outdir}/fermi_dwn_link", links)

        self._download(datadir=self.datadir)


