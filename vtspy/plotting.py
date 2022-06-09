import numpy as np

from fermipy.plotting import ROIPlotter, SEDPlotter

from gammapy.visualization import plot_spectrum_datasets_off_regions

import matplotlib.pyplot as plt

from scipy.stats import chi2
from scipy.stats import norm


def fermi_plotter(name, output, roi, config, subplot = None, **kwargs):
    """
    This returns various plots generated from fermipy.ROIPlotter and fermipy.SEDPlotter

    Args:
        name (str): a plot to show
            Options: ["sqrt_ts", "npred", "ts_hist", 
                      "data", "model", "sigma", 
                      "excess", "resid", "sed"]
            Default: config.yaml
        output (dict): output dictionaray generated from FermiAnalysis.analysis
        roi (GTanalysis.roi)
        config (GTanalysis.config)
        subplot: location of a plot
            Default: None
        **kwargs
    """
    kwargs.setdefault('graticule_radii', config['plotting']['graticule_radii'])
    kwargs.setdefault('label_ts_threshold',
                      config['plotting']['label_ts_threshold'])
    kwargs.setdefault('cmap', config['plotting']['cmap'])
    kwargs.setdefault('catalogs', config['plotting']['catalogs'])

    ymin = kwargs.get('ymin', None)
    showlnl = kwargs.get('showlnl', False)
    
    if subplot is None:
        fig = plt.figure(figsize=(14,6))
        subplot = "111"
        ax = plt.gca()
    else:
        ax = None

    if name == "sqrt_ts":
        sigma_levels = [3, 5, 7] + list(np.logspace(1, 3, 17))
        ROIPlotter(output['ts']['sqrt_ts'], roi=roi, **kwargs).plot(
            vmin=0, vmax=5, levels=sigma_levels, 
            cb_label='Sqrt(TS) [$\sigma$]', 
            interpolation='bicubic', subplot=subplot)
        ax = plt.gca()
        ax.set_title('Sqrt(TS)')

    if name == "npred":
        ROIPlotter(output['ts']['npred'], roi=roi, **kwargs).plot(
            vmin=0, cb_label='NPred [Counts]', interpolation='bicubic')
        ax = plt.gca()
        ax.set_title('NPred')

    if name == "ts_hist":
        ax = plot_ts_hist(output, subplot)
        ax.set_title('TS histogram')

    if name == "data":
        ROIPlotter(output['resid']['data'],roi=roi).plot(vmin=50,vmax=400,subplot=subplot,cmap='magma')
        ax = plt.gca()
        ax.set_title('Data')

    if name == "model":
        ROIPlotter(output['resid']['model'],roi=roi).plot(vmin=50,vmax=400,subplot=subplot,cmap='magma')
        ax = plt.gca()
        ax.set_title('Model')

    if name == "sigma":
        ROIPlotter(output['resid']['sigma'],roi=roi).plot(vmin=-5,vmax=5,levels=[-5,-3,3,5],subplot=subplot,cmap='RdBu_r')
        ax = plt.gca()
        ax.set_title('Significance')

    if name == "excess":
        ROIPlotter(output['resid']['excess'],roi=roi).plot(vmin=-100,vmax=100,subplot=subplot,cmap='RdBu_r')
        ax = plt.gca()
        ax.set_title('Excess')

    if name == "resid":
        ax = plot_sigma_hist(output, subplot)
        ax.set_title('Residual histogram')

    if name == "sed":
        SEDPlotter(output["sed"]).plot(showlnl=showlnl)
        ax = plt.gca()
        ax.set_ylim(ymin)

    return ax 

def plot_ts_hist(output, subplot=None):
    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    bins = np.linspace(0, 25, 101)

    data = np.nan_to_num(output['ts']['ts'].data.T)
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

    return ax

def plot_sigma_hist(output, subplot=None):
    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)

    nBins = np.linspace(-6, 6, 121)
    data = np.nan_to_num(output['resid']['sigma'].data)

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

    return ax

def veritas_plotter(name, output, **kwargs):

    if name == "flux":
        ax = plt.gca()
        output.plot(ax, sed_type="e2dnde", color="lightblue", label="1E S1218+304")
        output.plot_ts_profiles(ax=ax, sed_type="e2dnde");
        ax.legend()

    elif name == "sed":
        kwargs_spectrum = {**kwargs, "kwargs_model": {"color":"blue", "label":"Pwl"}, "kwargs_fp":{"color":"blue", "marker":"o", "label":"1ES 1218+304"}}
        kwargs_residuals = {"color": "blue", "markersize":4, "marker":'s', }
        ax_spec, ax_res = output.plot_fit(kwargs_spectrum=kwargs_spectrum)

def plot_ROI(veritas=None, fermi=None):
    plt.figure(figsize=(7, 7))
        
    if veritas is not None:
        ax = veritas._exclusion_mask.plot()
        veritas._on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="red")
        plot_spectrum_datasets_off_regions(ax=ax, datasets=veritas.datasets)
    
    if fermi is not None:
        if veritas is not None:
            geom = Map.create(npix=(150, 150), binsz=0.05, skydir=fermi.target.skydir, proj="CAR", frame="icrs")
            _, ax, _ = geom.plot()
            ax.add_patch(Patches.Rectangle((0, 0), 150, 150,  color="w"))
        #fermi._src_in_roi(ax)

    plt.show(block=False)