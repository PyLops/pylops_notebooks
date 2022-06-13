import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from typing import Tuple, List, Union


# Credit : https://github.com/polimi-ispl/deep_prior_interpolation/blob/master/utils/plotting.py


def clim(in_content: np.ndarray, ratio: float = 95) -> Tuple[float, float]:
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def explode_volume(volume: np.ndarray, t: int = None, x: int = None, y: int = None,
                   figsize: tuple = (8, 8), cmap: str = 'bone', clipval: tuple = None, p: int = 98,
                   tlim: tuple = None, xlim: tuple = None, ylim: tuple = None, 
                   labels : list = ('[s]', '[km]', '[km]'),
                   tlabel : str = 't',
                   ratio: tuple = None, linespec: dict = None, interp: str = None, title: str = '',
                   filename: str or Path = None, save_opts: dict = None) -> plt.figure:
    if linespec is None:
        linespec = dict(ls='-', lw=1, color='orange')
    nt, nx, ny = volume.shape
    t_label, x_label, y_label = labels
    
    t = t if t is not None else nt//2
    x = x if x is not None else nx//2
    y = y if y is not None else ny//2

    if tlim is None:
        t_label = "samples"
        tlim = (-0.5, volume.shape[0] - 0.5)
    if xlim is None:
        x_label = "samples"
        xlim = (-0.5, volume.shape[1] - 0.5)
    if ylim is None:
        y_label = "samples"
        ylim = (-0.5, volume.shape[2] - 0.5)
    
    # vertical lines for coordinates reference
    tline = (tlim[1] - tlim[0]) / nt * t + tlim[0] + 0.5
    xline = (xlim[1] - xlim[0]) / nx * x + xlim[0] + 0.5
    yline = (ylim[1] - ylim[0]) / ny * y + ylim[0] + 0.5
    
    # instantiate plots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    if ratio is None:
        wr = (nx, ny)
        hr = (ny, nx)
    else:
        wr = ratio[0]
        hr = ratio[1]
    opts = dict(cmap=cmap, clim=clipval if clipval is not None else clim(volume, p), aspect='auto', interpolation=interp)
    gs = fig.add_gridspec(2, 2, width_ratios=wr, height_ratios=hr,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    
    # central plot
    ax.imshow(volume[:, :, y], extent=[xlim[0], xlim[1], tlim[1], tlim[0]], **opts)
    ax.axvline(x=xline, **linespec)
    ax.axhline(y=tline, **linespec)
    
    # top plot
    ax_top.imshow(volume[t].T, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], **opts)
    ax_top.axvline(x=xline, **linespec)
    ax_top.axhline(y=yline, **linespec)
    ax_top.invert_yaxis()
    
    # right plot
    ax_right.imshow(volume[:, x], extent=[ylim[0], ylim[1], tlim[1], tlim[0]], **opts)
    ax_right.axvline(x=yline, **linespec)
    ax_right.axhline(y=tline, **linespec)
    
    # labels
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("x " + x_label)
    ax.set_ylabel(tlabel + " " + t_label)
    ax_right.set_xlabel("y " + y_label)
    ax_top.set_ylabel("y " + y_label)
    
    if filename is not None:
        if save_opts is None:
            save_opts = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
        plt.savefig(f"{filename}.{save_opts['format']}", **save_opts)