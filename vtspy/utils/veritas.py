import os
import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from gammapy.maps import Map, MapAxis
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    Models,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
)


from regions import CircleSkyRegion

import glob, re

from pathlib import Path

SCRIPT_DIR = os.environ.get('vtspy')

class VeritasUtils:
    def __init__(self, verbosity=True, overwrite=False, exclusion = "Hipparcos_MAG8_1997.dat" **kwargs):

        e_reco = MapAxis.from_energy_bounds(0.1, 100, 40, unit="TeV", name="energy")
        e_true = MapAxis.from_energy_bounds(0.1, 100, 40, unit="TeV", name="energy_true")
        self._veritas_energy_bin = {"E_tr": e_true, "E_rec": e_reco}
        
        self._anasum_files = glob.glob("./veritas/*anasum*")
        if len(self._anasum_files) == 0:
            print("[Error] No DL3 files are found.")
            return None 

        self._veritas_outdir = Path("veritas")
        self._veritas_outdir.mkdir(exist_ok=True)

        
        if self.verbosity:
            print("[Log][0/6] Initializing the VERITAS setup...")
            print("[Log][1/6] Importing VERITAS DL3 files...")
        self.veritas_datastore = DataStore.from_dir("./veritas/")
        self._veritas_obs_ids = self._find_obs_id()
        self._veritas_obs = self.veritas_datastore.get_observations(self._veritas_obs_ids)

        if self.verbosity:
            print("[Log][2/6] Defining the target and its ON region")
        
        if not(hasattr(self, "_target")):
            self._set_target_veritas()
        else:
            name = self._get_target_veritas()
            self.setTarget(name)
        
        self._set_on_region()
        
        if self.verbosity:
            print("[Log][3/6] Setting the exclusion region(s)...")
        self._set_exclusion_regions(exclusion=exclusion)

        if self.verbosity:
            print("[Log][4/6] Defining OFF regions and exporting on-off regions...")
        self._set_off_regions()

        if self.verbosity:
            print("[Log][5/6] Defining spectral model for the target (Powerlaw with index = 2)...")
        self.VeritasDatasets = self.setVeritasDatasets()

        if self.verbosity:
            print("[Log][6/6] Finalizing the Veritas setup", end='\r')

        if overwrite:
            self.VeritasDatasets.write(self._veritas_outdir / "veritas_dataset.yaml", overwrite=True)

        if self.verbosity:
            print("[Log][6/6] Completed.                  ")

    @property
    def veritas_obs_ids(self):
        return self._veritas_obs_ids
    
    @property
    def veritas_obs(self):
        return self._veritas_obs
    
    
    @staticmethod
    def brightSrcList(source = "Hipparcos_MAG8_1997.dat"):
        bright_sources = []
        with open(SCRIPT_DIR+"/vtspy/refdata/"+source) as f: 
            for line in f.readlines()[27:]:
                info = line.split()
                if len(info) == 5:
                    ra = info[0]
                    dec = info[1]
                    bright = info[3]

                    bright_sources.append([float(ra), float(dec), float(bright)])
        return np.asarray(bright_sources)

    def setVeritasDatasets(self, models=None, roi_cut=0.1):
        datasets = Datasets()
        for obs_id in self._veritas_obs_ids:
            filename = str(self._veritas_outdir / "pha_obs{}.fits".format(obs_id))
            datasets.append(SpectrumDatasetOnOff.from_ogip_files(filename))

        if models == None:
            models = self._default_model()
        
        if hasattr(self, "models"):
            models = self._roi_models(roi_cut=roi_cut)
        else:
            self._models = models

        for dataset in datasets:
            dataset.models = models
        return datasets

    def _find_obs_id(self):
        obs_id = []
        for file in self._anasum_files:
            obs_id.append(int(re.findall("([0-9]+)", file)[0]))
        return obs_id
    
    def _get_target_veritas(self):
        name = None

        for file in self._anasum_files:
            header = fits.open(file)[1].header

            if name == None:
                name = header['OBJECT']
            else:
                temp = header['OBJECT']
                if temp != name:
                    print("[Warning] Object names from input files are different from one another.")
        return name

    def _set_target_veritas(self):
        name = None
        ra = None
        dec = None

        for file in self._anasum_files:
            header = fits.open(file)[1].header

            if name == None:
                name = header['OBJECT']
            else:
                temp = header['OBJECT']
                if temp != name:
                    print("[Warning] Object names from input files are different from one another.")

            if ra == None:
                ra = header['RA_OBJ']
            else:
                temp = header['RA_OBJ']
                if temp != ra:
                    print("[Warning] RA values from input files are different from one another.")
            
            if dec == None:
                dec = header['DEC_OBJ']
            else:
                temp = header['DEC_OBJ']
                if temp != dec:
                    print("[Warning] DEC values from input files are different from one another.")       
        
        skydir = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        self._target = veritas_target(name, skydir)
        self._target_id = 0

    def _set_on_region(self, th2_cut = 0.008):

        on_region_radius = Angle(np.sqrt(th2_cut)*u.deg)
        self._on_region = CircleSkyRegion(center=self.target.skydir, radius=on_region_radius)
        
    def _set_exclusion_regions(self, exclusion = "Hipparcos_MAG8_1997.dat"):

        bright_sources = self.brightSrcList(source = exclusion)

        roi_cut = (abs(bright_sources[:,0]-self.target.skydir.ra.deg) <5) \
                * (abs(bright_sources[:,1]-self.target.skydir.dec.deg) <5) \
                * (bright_sources[:,2] < 7)
        
        bright_sources = bright_sources[roi_cut]
        
        excluded_regions = []
        for src_pos in bright_sources:
            excluded_regions.append(CircleSkyRegion(
                center=SkyCoord(src_pos[0], src_pos[1], unit="deg", frame="icrs"),
                radius=0.5 * u.deg,))

        self._exclusion_mask = Map.create(
            npix=(150, 150), binsz=0.05, skydir=self.target.skydir, proj="CAR", frame="icrs"
        )

        self._exclusion_mask.data = self._exclusion_mask.geom.region_mask(excluded_regions, inside=False)

    def _set_off_regions(self):

        datasets = Datasets()
        
        dataset_empty = SpectrumDataset.create(e_reco=self._veritas_energy_bin["E_rec"], 
                                               e_true=self._veritas_energy_bin["E_tr"], 
                                               region=self._on_region)

        dataset_maker = SpectrumDatasetMaker(containment_correction=False, selection=["counts", "exposure", "edisp"])

        bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=self._exclusion_mask)
        safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)
        
        for obs_id, obs in zip(self._veritas_obs_ids, self._veritas_obs):
            dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), obs)
            dataset_on_off = bkg_maker.run(dataset, obs)
            dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
            datasets.append(dataset_on_off)

        for dataset in datasets:
            dataset.to_ogip_files(outdir=self._veritas_outdir, overwrite=True)

    def _default_model(self):
        spectral_model = PowerLawSpectralModel(
            index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
        )

        model = SkyModel(spectral_model=spectral_model, name="source")

        return Models([model])

    def _roi_models(self, roi_cut=0.1):
        ref = self.models[self.target_id].spatial_model.position
        models = []
        for model in self.models:
            if model.name != 'galdiff' and model.name != 'isodiff':
                if ref.separation(model.spatial_model.position).deg < roi_cut:
                        models.append(model)
        return Models(models)

class veritas_target:
    def __init__(self, name, skydir):
        self._name = name
        self._skydir = skydir
        
    @property
    def name(self):
        return self._name
    @property
    def skydir(self):
        return self._skydir
    
