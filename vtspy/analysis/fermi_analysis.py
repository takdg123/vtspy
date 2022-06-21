import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from ..config import JointConfig
from ..utils import logger
from ..plotting import fermi_plotter

from astropy import units as u
from astropy.coordinates import SkyCoord

from gammapy.data import EventList
from gammapy.datasets import MapDataset, Datasets

from gammapy.irf import PSFMap, EDispMap
from gammapy.maps import Map, MapAxis, WcsGeom

from ..model import *

from gammapy.modeling.models import Models

from gammapy.modeling import Fit

from fermipy.gtanalysis import GTAnalysis

import fermipy.wcs_utils as wcs_utils
import fermipy.utils as fermi_utils

from regions import CircleSkyRegion
from pathlib import Path

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

class FermiAnalysis():
    """
    This is to perform a simple Fermi-LAT analysis and to prepare the joint-fit
    analysis. All fermipy.GTanalysis functions and attributes can be accessed 
    with the 'gta' arribute. e.g.,  

        fermi = FermiAnalysis()
        fermi.gta.optimize()

    All information about the state of the analysis (See fermipy documentation 
    for details) will be saved in the numpy array format (npy).
    
    Args:
        state_file (str): state filename (npy)
            Default: initial
        config_file (str): config filename (yaml)
            Default: config.yaml
        overwrite (bool): overwrite the state
            Default: False
        remove_weak_srcs (bool): remove sources with TS of nan or 0
            Default: False 
        construct_dataset (bool): construct dataset for the gammapy analysis
            Default: False
        verbosity (int)
        **kwargs: passed to fermipy.GTAnalysis module
    """

    def __init__(self, state_file = "initial", config_file='config.yaml', overwrite=False, remove_weak_srcs = False, construct_dataset = False, verbosity=True, **kwargs):
        
        self._verbosity = verbosity
        config = JointConfig.get_config(config_file=config_file).pop("fermi")

        self._logging = logger(self.verbosity)
        self._logging.info("Initialize the Fermi-LAT analysis.")
        
        self.gta = GTAnalysis(config, logging={'verbosity' : self.verbosity+1}, **kwargs)
        self._outdir = self.gta.config['fileio']['outdir']

        self._energy_bins = MapAxis.from_bounds(1e2, 1e5, nbin=6, interp="log", unit="MeV").edges

        if overwrite or not(os.path.isfile("./{}/initial.fits".format(self._outdir))):
            
            if overwrite:
                self._logging.info("Overwrite the Fermi-LAT setup.")
            else:
                self._logging.info("Initial setup and configuration are not found. Performing the data reduction...")
            
            self._logging.debug("Generate fermipy files.")
            self.gta.setup(overwrite=overwrite)

            self.gta.config["data"]["ltcube"] = f"{self._outdir}/ltcube_00.fits"
            
            self._logging.debug("Optimize the ROI.")
            self.gta.optimize()
            
            self._update_model()
            if remove_weak_srcs:
                self.remove_weak_srcs()
            
            self._logging.debug("Generate PSF.")
            generatePSF(self.gta.config)
            
            if construct_dataset:
                self.construct_dataset()

            self.save_state("initial", init=True, **kwargs)
            
            self._logging.info("The initial setup and configuration is saved [state_file = initial].")
        else:
            self._logging.info("The setup and configuration is found [state_file = {}]. Loading the configuration...".format(state_file))
            
            flag = self.load_state(state_file)

            if flag == -1:
                return

            if construct_dataset:
                self.construct_dataset()

        try:
            self.output = np.load(f"./{self._outdir}/output.npy", allow_pickle=True).item()
        except:
            self.output = {}

        self._test_model = {'Index' : 2.0, 'SpatialModel' : 'PointSource' }
        self._find_target()
        self._logging.info("Completed (Fermi-LAT initialization).")
        
    @property
    def target(self):
        """
        Return:
            fermipy.roi_model.Source
        """
        return self._target

    @property
    def target_name(self):
        """
        Return:
            str: target name
        """
        return self._target_name
    
    @property
    def target_id(self):
        """
        Return:
            int: target id
        """
        return self._target_id
    
    def print_association(self):
        """
        Print sources within ROI and their associations.
        """

        for i, src in enumerate(self.gta.roi.sources):
            if src.name == "isodiff" or src.name=="galdiff":
                continue
            
            self._logging.info(str(i)+") "+src.name+":"+str(src.associations[1:]))

    def print_target(self):
        """
        Print the target properties
        """
        self._logging.info(self.gta.roi.sources[self.target_id])

    def print_model(self):
        """
        Print source models within ROI
        """
        return self.gta.print_model(loglevel=40)

    def print_params(self, full_output=False):
        """
        Print parameters of sources within ROI
        """
        if full_output:
            return self.gta.print_params(True, loglevel=40)
        else:
            return self.gta.print_params(False, loglevel=40)

    
    @property
    def verbosity(self):
        """
        Return:
            int
        """
        return self._verbosity


    def peek_events(self):
        """
        Show event information
        """
        if not(hasattr(self, "_gammapy_events")):
            self._logging.error("Run FermiAnalysis.construct_dataset first.")
            return 

        self._gammapy_events.peek()
        
    def peek_irfs(self):
        """
        Show instrument response function (irf) information
        """
        if not(hasattr(self, "datasets")):
            self._logging.error("Run FermiAnalysis.construct_dataset first.")
            return

        e_true = self.datasets.edisp.edisp_map.geom.axes[1]
        e_reco = MapAxis.from_energy_bounds(
            e_true.edges.min(),
            e_true.edges.max(),
            nbin=len(e_true.center),
            name="energy",
        )

        edisp_kernel = self.datasets.edisp.get_edisp_kernel(energy_axis=e_reco)

        
        f, ax = plt.subplots(2,2, figsize=(10, 6))
        edisp_kernel.plot_bias(ax = ax[0][0])
        ax[0][0].set_xlabel(f"$E_\\mathrm{{True}}$ [MeV]")

        edisp_kernel.plot_matrix(ax = ax[0][1])
        self.datasets.psf.plot_containment_radius_vs_energy(ax = ax[1][0])
        self.datasets.psf.plot_psf_vs_rad(ax = ax[1][1])
        plt.tight_layout()

    def save_state(self, state_file, init=False):
        """
        Save the state
        
        Args:
            state_file (str): passed to fermipy.write_roi
            init (bool): check whether this is the initial analysis.
                Default: False
        """
        
        if (init==False) and (state_file == "initial"):
            self._logging.warning("The 'inital' state is overwritten. This is not recommended.")
            self._logging.warning("The original 'inital' state is archived in the '_orig' folder.")
            os.system(f"mkdir ./{self._outdir}/_orig")
            for file in glob.glob(f"./{self._outdir}/*initial*"):
                os.sytem(f"mv {file} ./{self._outdir}/_orig/")

        self.gta.write_roi(state_file, save_model_map=True)

    def load_state(self, state_file):
        """
        Load the state
        
        Args:
            state_file (str): passed to fermipy.write_roi
        """
        try:
            self.gta.load_roi(state_file)
        except:
            self._logging.error("The state file does not exist. Check the name again")
            return -1

    def set_target(self, target):
        """
        Set/change the target 
        
        Args:
            target (str or int): target name or id
        """
        if type(target)==int:
            self._target = self.gta.roi.sources[target]
            self._target_id = target
            self._logging.info(f"The target is set to {self.gta.roi.sources[target].name}")
            return
        elif type(target)==str:
            self._find_target(name=target)
            self._logging.info(f"The target is set to {src.name}")
        self._logging.warning("The entered target is not found. Check sources by using print_association.")
    
    def remove_weak_srcs(self):
        """
        Remove sources within ROI if they are too weak (TS < 0.1 or nan).
        """
        N = 0
        for src in self.gta.roi.sources:
            if src.name == "isodiff" or src.name=="galdiff":
                continue
            if np.isnan(src['ts']) or src['ts'] < 0.1:
                self.gta.delete_source(src.name)
                N+=1
        self._logging.info(f"{N} sources are deleted.")


    def fit(self, state_file="simple", pre_state=None, 
        free_all=False, free_target=True,
        remove_weak_srcs=False, fix_index=False, 
        min_ts=5, distance=3.0, optimizer = 'NEWMINUIT', 
        return_output=False, **kwargs):
        """
        Perform a simple fitting with various cuts
        
        Args:
            state_file (str): output state filename (npy)
                Default: simple
            free_all (bool): make the target's all parameters free
                Default: True
            free_all (bool): make all sources parameters free
                Default: False
            remove_weak_srcs (bool): remove sources with TS of nan or 0. This setting 
                will trigger another round of fit process after the first run.
                Default: False
            fix_index (bool): fix spectral shapes for sources for TS less than min_ts
                Default: False
            min_ts (int): minimum TS value for fixing a spectral shape
                Default: 5
            distance (float): parameters for sources outside of a certain distance 
                from the center are fixed, except for the normalization
                Default: 3.0 
            optimizer (str): either MINUIT or NEWMINUIT
                Default: NEWMINUIT
            return_output (bool): return the fitting result (dict)
                Default: False
            pre_state (str, optional): input state filename (npy). If not defined, starting from
                the current state.
                Default: None
            **kwargs: passed to fermipy.GTAnalysis.free_sources function

        Return
            dict: the output of the fitting when return_output is True 
        """
        if pre_state is not None:
            self.load_state(pre_state)

        if free_all:
            self.gta.free_sources(free=True, **kwargs) 
        else:
            self.gta.free_sources(free=False)

            self.gta.free_sources(free=True, distance=distance,  pars='norm', **kwargs)

            if not(fix_index):
                self.gta.free_sources(free=True, minmax_ts=[min_ts, None], **kwargs)
        
        if free_target:
            self.gta.free_sources_by_name(self.target_name, free=True, pars=None)

        o = self.gta.fit(optimizer=optimizer, verbosity=False)
        
        if remove_weak_srcs:
            self.remove_weak_srcs()
            o = self.gta.fit(optimizer=optimizer, verbosity=False)
        
        if o["fit_success"]:
            self._logging.info("Fit successfully.")
        else:
            self._logging.error("Fit failed.")

        self.save_state(state_file)
        if return_output:
            return o

    def analysis(self, jobs = ["ts", "resid", "sed"], 
        filename = "output", state_file="analyzed", **kwargs):
        """
        Perform various analyses: TS map, Residual map, and SED.
        
        Args:
            jobs (str or list): list of jobs, 'ts', 'resid', and/or 'sed'.
                Default: ['ts', 'resid', 'sed']
            filename (str): read and write the output
                Default: output
            state_file (str): output state filename (npy)
                Default: analyzed
            **kwargs: passed to GTanalysis.sed
        """

        try:
            output = np.load(f"./{self._outdir}/{filename}.npy", allow_pickle=True).item()
        except:
            output = {}
        
        model = kwargs.get("model", self._test_model)
        energy_bins = kwargs.get("energy_bins", [2.0,2.5,3.0,3.5,4.0,4.5,5.0])

        free = self.gta.get_free_param_vector()
        
        if type(jobs) == str:
            jobs = [jobs]

        if "ts" in jobs:
            o = self._ts_map(model=model)
            output['ts'] = o

        if "resid" in jobs:
            o = self._resid_dist(model=model)
            output['resid'] = o
            
        if "sed" in jobs:
            o = self._calc_sed(**kwargs)
            output['sed'] = o

        self.gta.set_free_param_vector(free)

        self.output = output
        np.save(f"./{self._outdir}/"+filename, output)

        self.save_state(state_file)

    def plot(self, output, filename="output", **kwargs):
        """
        Show various plots: TS map, Residual map, and SED.
        
        Args:
            output (str or list): list of plots to show
                Options: ["sqrt_ts", "npred", "ts_hist", "data", 
                "model", "sigma", "excess", "resid", "sed"]
            filename (str): read the output (from FermiAnalysis.analysis)
        """

        try:
            self._logging.info("Loading the output file...")
            self.output = np.load(f"./{self._outdir}/{filename}.npy", allow_pickle=True).item()
        except:
            self._logging.error("Run FermiAnalysis.analysis first.")
            return

        list_of_fig = ["sqrt_ts", "npred", "ts_hist", 
                        "data", "model", "sigma", "excess", "resid",
                        "sed"]

        if type(output) == str:
            output = [output]

        for o in output:
            if o not in list_of_fig:
                output.remove(o)

        if len(output) == 1:
            sub = "11"
        elif len(output) <= 3:
            sub = "1"+str(len(output))
            f = plt.figure(figsize=(4*len(output), 4))
        elif len(output) == 4:
            sub = "22"
            f = plt.figure(figsize=(8, 8))
        elif len(output) == 6:
            sub = "23"
            f = plt.figure(figsize=(12, 8))

        for i, o in enumerate(output):

            subplot = int(sub+f"{i+1}")
            ax = fermi_plotter(o, self, subplot=subplot, **kwargs)

        plt.tight_layout()
        

    def find_sources(self, state_file = "wt_new_srcs", re_fit=True, **kwargs):
        """
        Find sources within the ROI (using GTanalysis.find_sources). 
        
        Args:
            state_file (str): output state filename (npy)
                Default: wt_new_srcs
            re_fit (bool): re fit the ROI with new sources
            **kwargs: passed to fit.

        Return:
            dict: output of GTanalysis.find_sources
        """
        srcs = self.gta.find_sources(model=self._test_model, sqrt_ts_threshold=5.0,
                        min_separation=0.5)

        self._logging.info("{} sources are found. They will be added into the model list.".format(len(srcs["sources"])))

        if re_fit:
            self.fit(state_file=state_file, **kwargs)
        else:
            self.save_state(state_file)

        return srcs


    def construct_dataset(self, 
                        eventlist = "ft1_00.fits", 
                        exposure = "bexpmap_00.fits", 
                        psf = "gtpsf_00.fits"):

        """
        Construct a dataset for the gammapy analysis. 
        
        Args:
            eventlist (str): event list file (gtapp.maketime)
                Default: ft1_00.fits
            exposure (str): exposure map file (gtapp.gtexpmap)
            psf (str): psf file (gtapp.gtpsf)
        """

        self._logging.info("Loading the Fermi-LAT events...")
        counts = self._load_fermi_events(eventlist=eventlist)
        self._logging.info("Loading the Fermi-LAT IRFs...")
        irf = self._load_fermi_irfs(counts, exposure=exposure, psf=psf)
        self._logging.info("Loading the Fermi-LAT models...")
        models =  self._convert_model()
        self.datasets = MapDataset(
            name="fermi", models=models, counts=counts, exposure=irf["exposure"], psf=irf["psf"], edisp=irf["edisp"]
        )
        self._logging.info("Ready to perform a gammapy analysis.")

    def _find_target(self, name=None):
        if name is None:
            name = self.gta.config['selection']['target']

        flag = False
        for i, src in enumerate(self.gta.roi.sources):
            if src.name == "isodiff" or src.name=="galdiff":
                continue

            for n in src.associations:
                if n.replace(" ", "") == name:
                    self._target = self.gta.roi.sources[i]
                    self._target_name = self.gta.roi.sources[i].name
                    self._target_id = i
                    list_of_association = src.associations
                    flag = True
            if flag:
                break

        if flag:
            self._logging.info("The target, {}, is associated with {} source(s).".format(self.target_name, len(list_of_association)-1))
            self._logging.debug(list_of_association)
        else:
            self._logging.warning("The target name defined in the config file is not found.")
            self._target = self.gta.roi.sources[0]
            self._target_name = self.target.name
            self._target_id = 0

    def _ts_map(self, model):
        self._logging.info("Generating a TS map...")
        self.gta.free_sources(free=False)
        self.gta.free_sources(pars="norm")
        o = self.gta.tsmap('ts', model=model, write_fits=True, write_npy=True, make_plots=True)
        self._logging.info("Generating the TS map is completed.")
        return o

    def _resid_dist(self, model):
        self._logging.info("Generating a residual distribution...")
        self.gta.free_sources(free=False)
        self.gta.free_sources(pars="norm")
        o = self.gta.residmap('resid',model=model, write_fits=True, write_npy=True)
        self._logging.info("Generating the residual distribution is completed.")
        return o

    def _calc_sed(self, target=None, **kwargs):
        self._logging.info("Generating a SED... ")

        if target is None:
            target = self.target_name
        
        distance = kwargs.pop("distance", 3.0)

        loge_bins = kwargs.pop("loge_bins",  np.log10(self._energy_bins.value))
        
        self.gta.free_sources(free=False)
        self.gta.free_sources(skydir=self.gta.roi[target].skydir, distance=[distance], free=True)
        o = self.gta.sed(self.target.name, outfile='sed.fits', bin_index=2.2, loge_bins=loge_bins, write_fits=True, write_npy=True, **kwargs)
        self._logging.info("Generating the SED is completed.")
        return o

    def _lightcurve(self, target=None, **kwargs):

        self._logging.info("Generating a light curve...")

        if target is None:
            target = self.target_name
    
        free_radius = kwargs.pop("free_radius", 3.0)
        
        self.gta.free_sources(free=False)
        self.gta.free_sources(pars="norm")

        o = self.gta.lightcurve(target, free_radius=free_radius, multithread=True, 
            nthread=4, use_scaled_srcmap=True, **kwargs)

        self._logging.info("Generating the lightcurve is completed.")
        return o

    def _load_fermi_events(self, eventlist = "ft1_00.fits"):

        self._gammapy_events = EventList.read(Path(self._outdir)/ f"{eventlist}")

        glon = self.gta.config['selection']['glon']
        glat = self.gta.config['selection']['glat']
        energy_bins = np.logspace(2, 5.5, 8)* u.MeV
        src_pos = SkyCoord(glon, glat, unit="deg", frame="galactic")
        energy_axis = MapAxis.from_edges(energy_bins, name="energy", unit="MeV", interp="log")
        counts = Map.create(skydir=src_pos, width=self.gta.config['binning']['roiwidth'], 
            proj=self.gta.config['binning']['proj'], binsz=self.gta.config['binning']['binsz'], 
            frame='galactic', axes=[energy_axis], dtype=float)
        counts.fill_by_coord({"skycoord": self._gammapy_events.radec, "energy": self._gammapy_events.energy})
        
        return counts

    def _load_fermi_irfs(self, counts, exposure = "bexpmap_00.fits", psf = "gtpsf_00.fits"):
        expmap = Map.read(Path(self._outdir) / f"{exposure}")
        axis = MapAxis.from_nodes(
            counts.geom.axes[0].center, 
            name="energy_true",
            unit="MeV", 
            interp="log"
        )

        irf = {}

        geom = WcsGeom(wcs=counts.geom.wcs, npix=counts.geom.npix, axes=[axis])
        exposure = expmap.interp_to_geom(geom)
        irf['exposure'] = exposure
        
        # PSF
        psf = PSFMap.read(Path(self._outdir) / f"{psf}", format="gtpsf")
        irf['psf'] = psf

        # Energy dispersion
        e_true = exposure.geom.axes["energy_true"]
        edisp = EDispMap.from_diagonal_response(energy_axis_true=e_true)
        irf['edisp'] = edisp

        return irf

    def _convert_model(self):
        gammapy_models = []
        for src in self.gta.roi.sources:
            
            if src.name == "isodiff":
                gammapy_models.append(fermi_isotropic_diffuse_bkg(self.gta.config, src))
            elif src.name == "galdiff":
                gammapy_models.append(fermi_galactic_diffuse_bkg(self.gta.config, src))
            else:
                gammapy_models.append(fermipy2gammapy(self.gta.like, src))
            
            self._logging.debug(f"A source model for {src.name} is converted.")

        return Models(gammapy_models)
      
