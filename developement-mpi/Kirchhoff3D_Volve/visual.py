import numpy as np
import matplotlib.pyplot as plt


def plotting_style():
    plt.style.use('default')

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 18

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def clim(in_content, ratio=95):
    """Clipping based on percentiles
    """
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def explode_volume(volume, t=None, x=None, y=None,
                   figsize=(8, 8), cmap='bone', clipval=None, p=98,
                   tlim=None, xlim=None, ylim=None,
                   tcrop=None, xcrop=None, ycrop=None,
                   labels=('[s]', '[km]', '[km]'),
                   tlabel='t', xlabel='x', ylabel='y',
                   ratio=None, linespec=None, interp=None, title='',
                   filename=None, save_opts=None):
    """Display 3D volume

    Display 3D volume in exploding format (three slices)

    Credits : https://github.com/polimi-ispl/deep_prior_interpolation/blob/master/utils/plotting.py

    Parameters
    ----------
    volume : :obj:`numpy.ndarray`
        3D volume of size ``(nt, nx, ny)``
    t : :obj:`int`, optional
        Slicing index along time axis
    x : :obj:`int`, optional
        Slicing index along x axis
    y : :obj:`int`, optional
        Slicing index along y axis
    figsize : :obj:`bool`, optional
        Figure size
    cmap : :obj:`str`, optional
        Colormap
    clipval : :obj:`tuple`, optional
        Clipping min and max values
    p : :obj:`str`, optional
        Percentile of max value (to be used if ``clipval=None``)
    tlim : :obj:`tuple`, optional
        Limits of time axis in volume
    xlim : :obj:`tuple`, optional
        Limits of x axis in volume
    ylim : :obj:`tuple`, optional
        Limits of y axis in volume
    tlim : :obj:`tuple`, optional
        Limits of cropped time axis to be visualized
    xlim : :obj:`tuple`, optional
        Limits of cropped x axis to be visualized
    ylim : :obj:`tuple`, optional
        Limits of cropped y axis to be visualized
    labels : :obj:`bool`, optional
        Labels to add to axes as suffixes
    tlabels : :obj:`bool`, optional
        Label to use for time axis
    xlabels : :obj:`bool`, optional
        Label to use for x axis
    ylabels : :obj:`bool`, optional
        Label to use for y axis
    ratio : :obj:`float`, optional
        Figure aspect ratio (if ``None``, inferred from the volume sizes directly)
    linespec : :obj:`dict`, optional
        Specifications for lines indicating the selected slices
    interp : :obj:`str`, optional
        Interpolation to apply to visualization
    title : :obj:`str`, optional
        Figure title
    filename : :obj:`str`, optional
        Figure full path (if provided the figure is saved at this path)
    save_opts : :obj:`dict`, optional
        Additonal parameters to be provided to :func:`matplotlib.pyplot.savefig`

    Returns
    -------
    fig : :obj:`matplotlib.pyplot.Figure`
        Figure handle
    axs : :obj:`matplotlib.pyplot.Axis`
        Axes handes

    """
    if linespec is None:
        linespec = dict(ls='-', lw=1.5, color='#0DF690')
    nt, nx, ny = volume.shape
    t_label, x_label, y_label = labels
    
    t = t if t is not None else nt//2
    x = x if x is not None else nx//2
    y = y if y is not None else ny//2

    if tlim is None:
        t_label = "samples"
        tlim = (-0.5, nt - 0.5)
    if xlim is None:
        x_label = "samples"
        xlim = (-0.5, nx - 0.5)
    if ylim is None:
        y_label = "samples"
        ylim = (-0.5, ny - 0.5)
    
    # vertical lines for coordinates reference
    dt, dx, dy = (tlim[1] - tlim[0]) / nt, (xlim[1] - xlim[0]) / nx, (ylim[1] - ylim[0]) / ny
    tline = dt * t + tlim[0] + 0.5 * dt
    xline = dx * x + xlim[0] + 0.5 * dx
    yline = dy * y + ylim[0] + 0.5 * dy
    
    # instantiate plots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontweight='bold', y=0.95)
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
    if xcrop is not None:
        ax.set_xlim(xcrop)
    if tcrop is not None:
        ax.set_ylim(tcrop[1], tcrop[0])
    
    # top plot
    ax_top.imshow(volume[t].T, extent=[xlim[0], xlim[1], ylim[1], ylim[0]], **opts)
    ax_top.axvline(x=xline, **linespec)
    ax_top.axhline(y=yline, **linespec)
    ax_top.invert_yaxis()
    if xcrop is not None:
        ax_top.set_xlim(xcrop)
    if ycrop is not None:
        ax_top.set_ylim(ycrop[1], ycrop[0])

    # right plot
    ax_right.imshow(volume[:, x], extent=[ylim[0], ylim[1], tlim[1], tlim[0]], **opts)
    ax_right.axvline(x=yline, **linespec)
    ax_right.axhline(y=tline, **linespec)
    if ycrop is not None:
        ax_right.set_xlim(ycrop)
    if tcrop is not None:
        ax_right.set_ylim(tcrop[1], tcrop[0])

    # labels
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(xlabel + " " + x_label)
    ax.set_ylabel(tlabel + " " + t_label)
    ax_right.set_xlabel(ylabel + " " + y_label)
    ax_top.set_ylabel(ylabel + " " + y_label)
    
    if filename is not None:
        if save_opts is None:
            save_opts = {'format': 'png', 'dpi': 150, 'bbox_inches': 'tight'}
        plt.savefig(f"{filename}.{save_opts['format']}", **save_opts)
       
    return fig, (ax, ax_right, ax_top)