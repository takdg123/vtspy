---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: vtspy
  language: python
  name: vtspy
---

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

from vtspy import *
```

## Step 1. Generate a configuration file for Fermipy

```{code-cell} ipython3
ls ./veritas
```

```{code-cell} ipython3

config = JointConfig(files="./veritas/")
# or FermipyConfig(files="./veritas/95127.anasum.fits")
```

```{code-cell} ipython3
config.print_config()
```

## Step 2. Download Fermi-LAT data

```{code-cell} ipython3
dwn = DownloadFermiData(verbosity=1)
```

## Step 3. Analyze the Ferrmi-LAT data

+++

### Initiate fermipy

```{code-cell} ipython3
fermi = FermiAnalysis()
```

```{code-cell} ipython3
fermi.print_model()
```

### Check whether our source is in the list

```{code-cell} ipython3
fermi.print_association()
```

### Perform a simple analysis

```{code-cell} ipython3
o = fermi.fit(return_output=True)
```

```{code-cell} ipython3
fermi.print_model()
```

### Remove weak sources

```{code-cell} ipython3
fermi.remove_weak_srcs()
fermi.fit()
```

```{code-cell} ipython3
fermi.print_model()
```

```{code-cell} ipython3
fermi.print_params()
```

### Check TS distribution

```{code-cell} ipython3
fermi.analysis(jobs=["ts"])
```

```{code-cell} ipython3
fermi.plot(["sqrt_ts", "ts_hist"])
```

### Check resid distribution

```{code-cell} ipython3
fermi.analysis(jobs=["resid"])
```

```{code-cell} ipython3
fermi.plot(["sigma", "resid"])
```

### Calculate SED

```{code-cell} ipython3
fermi.analysis(jobs=["sed"])
```

```{code-cell} ipython3
fermi.plot("sed")
```

## Step 4. Construct dataset for the joint-fit analysis

```{code-cell} ipython3
fermi.construct_dataset()
```

### Peek events and irfs

```{code-cell} ipython3
fermi.peek_events()
```

```{code-cell} ipython3
fermi.peek_irfs()
```

### One can fit the Fermi-LAT datasets with gammapy (which takes long time)

```{code-cell} ipython3
from gammapy.modeling import Fit
```

```{code-cell} ipython3
gfit = Fit()
gfit.run(fermi.datasets)
```

```{code-cell} ipython3
fermi.plot("sed", erg=True, units="GeV", show_flux_points=False, show_band=True, label="fermipy")
gamma_result.plot([100*u.MeV, 300*u.GeV], sed_type="e2dnde", label="gammapy")
gamma_result.plot_error([100*u.MeV, 300*u.GeV], sed_type="e2dnde")
plt.legend(fontsize=15)
```

```{code-cell} ipython3

```
