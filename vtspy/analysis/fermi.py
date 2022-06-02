from fermipy.gtanalysis import GTAnalysis
from fermipy.plotting import ROIPlotter, SEDPlotter
from fermipy.roi_model import Source

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.stats import norm

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def generatePSF(config):  
    from GtApp import GtApp
    gtpsf = GtApp('gtpsf')
    workdir = config['fileio']['workdir']
    gtpsf["expcube"] = '{}/ltcube_00.fits'.format(workdir)
    gtpsf["outfile"] = '{}/gtpsf_00.fits'.format(workdir)
    gtpsf["irfs"] = config['gtlike']['irfs']
    gtpsf['evtype'] = config['selection']['evtype']
    gtpsf['ra'] = config['selection']['ra']
    gtpsf['dec'] = config['selection']['dec']
    gtpsf['emin'] = config['selection']['emin']
    gtpsf['emax'] = config['selection']['emax']
    gtpsf['chatter'] = 0
    gtpsf.run()

class FermiAnalysis(GTAnalysis):
    def __init__(self, file='config.yaml', roi = "initial", overwrite=False, verbosity=True, removeNaN = False, **kwargs):
        
        loglevel = 3 if verbosity==2 else 1 
        self.verbosity = verbosity

        super().__init__(file, logging={'verbosity' : loglevel}, **kwargs)
        if self.verbosity: 
            print("[Log] Initializing the Fermi-LAT analysis...")
        if overwrite or not(os.path.isfile("./{}/initial.fits".format(self.config['fileio']['outdir']))):
            if self.verbosity: 
                if overwrite:
                    print("[Log] Overwriting the Fermi-LAT setup...")
                else:
                    print("[Log] Initial setup and configuration are not found. Performing the data reduction...")
            self.setup(overwrite=overwrite, loglevel=50)
            self.optimize(loglevel=40 if verbosity else 10)
            if removeNaN:
                self.RemoveNaN()
            generatePSF(self.config)
            self.saveStatus("initial")
            if self.verbosity: 
                print("[Log] Initial setup and configuration are saved [roi = initial].")
        else:
            if self.verbosity: 
                print("[Log] Initial setup and configuration [roi = {}] are found. Loading the configuration...".format(roi))
            
            self.loadStatus(roi)

        try:
            self.output = np.load("./fermi/output.npy", allow_pickle=True).item()
        except:
            self.output = {}

        if self.verbosity: 
            print("[Log] Initialization of Fermi-LAT has been completed.")
        
        self._target = self.roi.sources[0]
        self._target_id = 0

    def saveStatus(self, name):
        self.write_roi(name, save_model_map=True, loglevel=self.loglevel)

    def loadStatus(self, name):
        self.load_roi(name)

    @property
    def target(self):
        return self._target
    
    @property
    def target_id(self):
        return self._target_id
    
    @property
    def printAssociation(self):
        i = 1
        for src in self.roi.sources:
            
            if src.name == "isodiff" or src.name=="galdiff":
                continue
            
            print(i, ":", src.name)
            print("\t", src.associations)
            i+=1
    @property
    def printTaget(self):
        print(self.roi.sources[self.target_id])

    def RemoveNaN(self):
        for src in self.roi.sources:
            if np.isnan(src['ts']):
                self.delete_source(src.name)
    
    def setTarget(self, target):
        if type(target)==int:
            self._target = self.roi.sources[target-1]
            self._target_id = target-1
            print("[Log] A target is set to", self.roi.sources[target-1].name)
            return
        elif type(target)==str:
            i = 0
            for src in self.roi.sources:
                if src.name == "isodiff" or src.name=="galdiff":
                    continue
                elif target in src.associations:
                    self._target = self.roi.sources[i]
                    self._target_id = i
                    print("[Log] A target is set to", src.name)
                    return
                else:
                    i+=1
        print("[Warning] The entered target is not found. Check sources by using printAssociation.")


    @property
    def printModel(self):
        return self.print_model(loglevel=40)

    def createSource(self, name="source"):
        src = Source(name, {'ra': self.config['selection']['ra'], 'dec': self.config['selection']['dec']})
        print(src)
        return src

    def addSource(self, src):
        return self.add_source(src.name, src)

    def simpleAnalysis(self, free_all=False, min_ts=5, roi_cut=None, fix_index=False, optimizer = 'MINUIT', output=False):
        
        if free_all:
            self.free_sources(free=True, loglevel=self.loglevel)    
        else:
            self.free_sources(free=False, loglevel=self.loglevel)

            self.free_sources(free=True, distance=roi_cut,  pars='norm', loglevel=self.loglevel)

            if not(fix_index):
                self.free_sources(free=True, minmax_ts=[min_ts, None], loglevel=self.loglevel)
        
        o = self.fit(optimizer=optimizer, verbosity=self.verbosity)
        
        self.saveStatus("simple")
        if output:
            return o


    def makeOutput(self, jobs = ["ts", "resid", "sed", "lc"], bins = 1, bin_def="time", free_radius=3.0):

        try:
            output = np.load("./fermi/output.npy", allow_pickle=True).item()
        except:
            output = {}
        
        model = {'Index' : 2.0, 'SpatialModel' : 'PointSource' }

        free = self.get_free_param_vector()

        target = self.target.name
        
        done = 0
        if "ts" in jobs:
            if self.verbosity: 
                print("[Log] Working on the TS map... [{}/{}]".format(done, len(jobs)), end='\r')
            self.free_sources(free=False)
            self.free_sources(pars="norm")
            ts_output = self.tsmap('ts',model=model, write_fits=True, write_npy=True)
            output['tsmap'] = ts_output
            done+=1
            if self.verbosity: 
                print("[Log] Completed [{}/{}].                      ".format(done, len(jobs)), end='\r')
            

        if "resid" in jobs:
            if self.verbosity: 
                print("[Log] Working on the residual map... [{}/{}]".format(done, len(jobs)), end='\r')
            self.free_sources(free=False)
            self.free_sources(pars="norm")
            resid_output = self.residmap('resid',model=model, write_fits=True, write_npy=True)
            output['resid'] = resid_output
            done+=1
            if self.verbosity: 
                print("[Log] Completed [{}/{}].                      ".format(done, len(jobs)), end='\r')

        if "sed" in jobs:
            if self.verbosity: 
                print("[Log] Working on the spectral energy distribution (SED)... [{}/{}]".format(done, len(jobs)), end='\r')
            
            self.free_sources(free=False)
            self.free_sources(skydir=self.roi[target].skydir, distance=[free_radius], free=True)
            sed_output = self.sed(target, outfile='sed.fits', bin_index=2.2, loge_bins=[2.0,2.5,3.0,3.5,4.0,4.5,5.0], write_fits=True, write_npy=True)
            output['sed'] = sed_output
            done+=1
            if self.verbosity: 
                print("[Log] Completed [{}/{}].                                                ".format(done, len(jobs)), end='\r')

        if "lc" in jobs:
            if self.verbosity: 
                print("[Log] Working on the residual map... [{}/{}]".format(done, len(jobs)), end='\r')
            
            if bin_def == "time":
                lcbin = {"binsz": bins*86400.}
            elif bin_def == "num":
                lcbin = {"nbins": bins}
            elif bin_def == "edge":
                lcbin = {"time_bins": bins}

            self.free_sources(free=False)
            lc_output = self.lightcurve(target, free_radius=free_radius, multithread=False, nthread=4, use_scaled_srcmap=True, **lcbin)

            output['lc'] = lc_output
            done+=1
            if self.verbosity: 
                print("[Log] Completed [{}/{}].                               ".format(done, len(jobs)), end='\r')

        self.set_free_param_vector(free)

        self.output = output
        np.save("./fermi/output", output)

    def plotTSMap(self, show=["sqrt_ts", "ts_hist"]):
        try:
            self.output = np.load("./fermi/output.npy", allow_pickle=True).item()
        except:
            print("[Error] Please run makeOutput(jobs=['ts']) first.")
            return

        fig = plt.figure(figsize=(14,6))

        size = len(show)
        loc = 1

        if "sqrt_ts" in show:
            subplot = int("1"+str(size)+str(loc))
            ROIPlotter(self.output['tsmap']['sqrt_ts'], roi=self.roi).plot(levels=[0,3,5,7],vmin=0,vmax=5,subplot=subplot,cmap='magma')
            plt.gca().set_title('Sqrt(TS)')
            loc+=1

        if "npred" in show:
            subplot = int("1"+str(size)+str(loc))
            ROIPlotter(self.output['tsmap']['npred'], roi=self.roi).plot(vmin=0,vmax=100,subplot=subplot,cmap='magma')
            plt.gca().set_title('NPred')
            loc+=1

        if "ts_hist" in show:
            subplot = int("1"+str(size)+str(loc))
            self._ts_hist(subplot)
            plt.gca().set_title('TS histogram')
            loc+=1

        plt.show(block=False)

    def plotResidMap(self, show=["sigma", "hist"]):
        try:
            self.output = np.load("./fermi/output.npy", allow_pickle=True).item()
        except:
            print("[Error] Please run makeOutput(jobs=['resid']) first.")
            return

        size = len(show)
        if size > 3:
            print("[Error] The number of input panels is too large; len(show)>3). Use two canvases." )
            return

        loc = 1

        fig = plt.figure(figsize=(14,6))
        if "data" in show:
            subplot = int("1"+str(size)+str(loc))
            ROIPlotter(self.output['resid']['data'],roi=self.roi).plot(vmin=50,vmax=400,subplot=121,cmap='magma')
            plt.gca().set_title('Data')
            loc+=1

        if "model" in show:
            subplot = int("1"+str(size)+str(loc))
            ROIPlotter(self.output['resid']['model'],roi=self.roi).plot(vmin=50,vmax=400,subplot=122,cmap='magma')
            plt.gca().set_title('Model')
            loc+=1

        if "sigma" in show:
            subplot = int("1"+str(size)+str(loc))
            ROIPlotter(self.output['resid']['sigma'],roi=self.roi).plot(vmin=-5,vmax=5,levels=[-5,-3,3,5],subplot=121,cmap='RdBu_r')
            plt.gca().set_title('Significance')
            loc+=1

        if "excess" in show:
            subplot = int("1"+str(size)+str(loc))
            ROIPlotter(self.output['resid']['excess'],roi=self.roi).plot(vmin=-100,vmax=100,subplot=122,cmap='RdBu_r')
            plt.gca().set_title('Excess')
            loc+=1

        if "hist" in show:
            subplot = int("1"+str(size)+str(loc))
            self._sigma_hist(subplot)
            plt.gca().set_title('Residual histogram')
            loc+=1

        plt.show(block=False)  


    def plotSED(self, ylim=None, showlnl=False):
        try:
            self.output = np.load("./fermi/output.npy", allow_pickle=True).item()
        except:
            print("[Error] Please run makeOutput(jobs=['sed']) first.")
            return

        fig = plt.figure(figsize=(6,4))
        SEDPlotter(self.output["sed"]).plot(showlnl=showlnl)
        plt.gca().set_ylim(ylim)

        plt.show(block=False)            

    def _ts_hist(self, subplot):
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)
        bins = np.linspace(0, 25, 101)

        data = np.nan_to_num(self.output['tsmap']['ts'].data.T)
        data[data > 25.0] = 25.0
        data[data < 0.0] = 0.0
        n, bins, patches = ax.hist(data.flatten(), bins, density=True,
                                   histtype='stepfilled',
                                   facecolor='green', alpha=0.75)

        ax.plot(bins, 0.5 * chi2.pdf(bins, 1.0), color='k',
                label=r"$\chi^2_{1} / 2$")
        ax.set_yscale('log')
        ax.set_ylim(1E-4)
        ax.legend(loc='upper right', frameon=False)

        # labels and such
        ax.set_xlabel('TS')
        ax.set_ylabel('Probability')

    def _sigma_hist(self, subplot):
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)

        nBins = np.linspace(-6, 6, 121)
        data = np.nan_to_num(self.output['resid']['sigma'].data)

        # find best fit parameters
        mu, sigma = norm.fit(data.flatten())
        
        # make and draw the histogram
        data[data > 6.0] = 6.0
        data[data < -6.0] = -6.0

        n, bins, patches = ax.hist(data.flatten(), nBins, density=True,
                                   histtype='stepfilled',
                                   facecolor='green', alpha=0.75)
        # make and draw best fit line
        y = norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, 'r--', linewidth=2, label="Best-fit")
        y = norm.pdf(bins, 0.0, 1.0)
        ax.plot(bins, y, 'k', linewidth=1, label=r"$\mu$ = 0, $\sigma$ = 1")

        # labels and such
        ax.set_xlabel(r'Significance ($\sigma$)')
        ax.set_ylabel('Probability')
        paramtext = 'Gaussian fit:\n'
        paramtext += '$\\mu=%.2f$\n' % mu
        paramtext += '$\\sigma=%.2f$' % sigma
        ax.text(0.05, 0.95, paramtext, verticalalignment='top',
                horizontalalignment='left', transform=ax.transAxes)

        ax.legend()