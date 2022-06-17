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

## Step 1. Generate a configuration file

```{code-cell} ipython3
from vtspy import *
```

```{code-cell} ipython3
ls ./veritas
```

```{code-cell} ipython3
config = JointConfig(files="./veritas/")
```

```{code-cell} ipython3
config.print_config()
```

## Step 2. Analyze the VERITAS data

```{code-cell} ipython3
from vtspy.analysis import VeritasAnalysis

veritas = VeritasAnalysis(overwrite=True)
```

### Plot ON- and OFF- regions

```{code-cell} ipython3
veritas.plot("roi")
```

### Peek dataset

```{code-cell} ipython3
veritas.peek_dataset()
```

### Apply additional cuts

```{code-cell} ipython3
veritas.construct_dataset(eff_cut=20, bias_cut=20)
```

```{code-cell} ipython3
veritas.peek_dataset()
```

### Perform fit and do high-level analyses

```{code-cell} ipython3
veritas.fit(model="PowerLaw")
veritas.plot("fit")
print(veritas.fit_results.total_stat)
```

```{code-cell} ipython3
veritas.analysis()
```

### Plot the results

```{code-cell} ipython3
veritas.plot("flux")
```

```{code-cell} ipython3
veritas.plot("sed")
```

```{code-cell} ipython3
veritas.print_flux()
```

### Generate light curve

```{code-cell} ipython3
veritas.analysis(jobs="lc", nbins=2)
```

```{code-cell} ipython3
veritas.plot("lc")
```

```{code-cell} ipython3
veritas.print_lightcurve()
```
