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
```

```{code-cell} ipython3
from vtspy import *
```

## Step 1. Load Fermi-LAT and VERITAS datasets

```{code-cell} ipython3
joint = JointAnalysis(fermi="analyzed", veritas="analyzed")
```

### Check datasets and models

```{code-cell} ipython3
joint.print_datasets()
```

```{code-cell} ipython3
joint.print_models()
```

### Check a global SED before the fit

```{code-cell} ipython3
joint.sed_plot()
```

### Change a spectral model

```{code-cell} ipython3
joint.change_model("logparabola", optimize=True, method="flux")
joint.sed_plot()
```

## Step 2. Run a joint-fit analysis

```{code-cell} ipython3
from astropy.table import Table
sed_tab = Table.read('./fermi/sed.fits', hdu=1)
```

### Check a global SED after the fit

```{code-cell} ipython3
joint.sed_plot(show_flux_points=True)
```

## Bonus. Test the 'agnpy' model

```{code-cell} ipython3
from vtspy.model import default_model
```

```{code-cell} ipython3
agnpy = default_model("agnpy", redshift=0.182)
```

### Check initial parameters

```{code-cell} ipython3
agnpy.parameters.to_table()
```

```{code-cell} ipython3
agnpy.plot([100 * u.MeV, 30 * u.TeV], sed_type="e2dnde", label="AGN", color="r")
joint.sed_plot(show_flux_points=True)
```

### Change a spectral model

```{code-cell} ipython3
joint.change_model(agnpy, optimize=True, method="flux")
joint.sed_plot(show_flux_points=True)
```

```{code-cell} ipython3
joint.print_models()
```

### Fit the data

```{code-cell} ipython3
joint.fit()
```

```{code-cell} ipython3
joint.sed_plot(show_flux_points=True)
```

```{code-cell} ipython3
joint.print_models()
```

```{code-cell} ipython3

```
