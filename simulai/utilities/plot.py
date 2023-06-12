import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import ceil

def anim_2D(data, title, xlabel='NX', ylabel='NY', args={}, n_columns=None, max_tstep=-1, interval=100, min_val=None, max_val=None, figsize=None, fig=None, ax=None):
    """
    Animate 2D data.

    Parameters
    ----------
    data : list of 2D arrays
        List of 2D arrays to animate.
    title : list of strings
        List of titles for each 2D array.
    xlabel : string, optional
        Label for the x-axis. The default is 'NX'.
    ylabel : string, optional
        Label for the y-axis. The default is 'NY'.
    args : dict, optional
        Additional arguments to pass to imshow. The default is {}.
    n_columns : int, optional
        Number of columns in the plot. The default is None.
    max_tstep : int, optional
        Maximum time step to animate. The default is -1.
    interval : int, optional
        Interval between frames in milliseconds. The default is 100.

    Returns
    -------
    anim : matplotlib animation
        Animation of the plot.
    """
    
    if n_columns is None or n_columns > len(data):
        n_columns = len(data)
        
    if max_tstep is None or max_tstep < 0:
        max_tstep = data[0].shape[0]
    i=0
    fig, ax, im = plot_2D(data, title, xlabel, ylabel, args, n_columns, i, fig, ax, min_val, max_val, figsize)
    
    def animate(i):
        for j, imi in enumerate(im):
            imi.set_array(data[j][i,...].T[::-1])
    
    plt.tight_layout()
    plt.close()
    anim = FuncAnimation(fig, animate, interval=interval, frames=max_tstep)
    return anim

def plot_2D(data, title, xlabel='NX', ylabel='NY', args={}, n_columns=None, tstep=-1, fig=None, ax=None, min_val=None, max_val=None, figsize=None):
    """
    Plot 2D data.

    Parameters
    ----------
    data : list of 2D arrays
        List of 2D arrays to plot.
    title : list of strings
        List of titles for each 2D array.
    xlabel : string, optional   
        Label for the x-axis. The default is 'NX'.
    ylabel : string, optional
        Label for the y-axis. The default is 'NY'.
    args : dict, optional
        Additional arguments to pass to imshow. The default is {}.
    n_columns : int, optional
        Number of columns in the plot. The default is None.
    tstep : int, optional
        Time step to plot. The default is -1.
    fig : matplotlib figure, optional
        Figure to plot on. The default is None.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None.
    min_val : float, optional
        Minimum value for the colorbar. The default is None.
    max_val : float, optional
        Maximum value for the colorbar. The default is None.

    Returns
    -------
    fig : matplotlib figure
        Figure with the plot.
    ax : matplotlib axis
        Axis with the plot.
    im : matplotlib image
        Image of the plot.
    """
    if n_columns is None or n_columns > len(data):
        n_columns = len(data)
    n_data = len(data)
    n_rows = int(np.ceil(n_data/n_columns))

    if fig is None and ax is None:
        if figsize is None:
            figsize = (5.0*n_columns, n_rows*5.3)
        fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize)
        
        if n_columns == 1:
            ax = np.array([ax]).T
            
    if n_rows == 1: ax = [ax]

    if min_val is None:
        min_val = [None]*len(data)
    if max_val is None:
        max_val = [None]*len(data)
    
    vv = 0
    for i in  np.arange(n_rows):
        for j in np.arange(n_columns):

            if min_val[vv] is None:
                min_val[vv] = np.amin(data)
            if max_val[vv] is None:
                max_val[vv] = np.amax(data)
            vv += 1

    vv = 0
    im = []
    for i in  np.arange(n_rows):
        for j in np.arange(n_columns):
            if j == 0:
                ax[i][j].set_ylabel(ylabel, fontsize=12)
    
            ax[i][j].set_xlabel(xlabel, fontsize=12)
            ax[i][j].tick_params(axis='x', labelsize=12)
            ax[i][j].tick_params(axis='y', labelsize=12)

            if vv < n_data:
                im_= ax[i][j].imshow(data[vv][tstep,...], vmin=min_val[vv], vmax=max_val[vv],  **args)
                im.append(im_)
                cbar = fig.colorbar(im[vv], ax=ax[i][j], location='top', fraction=0.046, pad=0.04)
                cbar.ax.set_xlabel(f'{title[vv]}', fontsize=12)
            else:
                ax[i][j].axis('off')
            vv += 1
    
    plt.tight_layout()
    return fig, ax, im

def plot3(fom, rom, i=0, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,3, figsize=(13,5.3))

    
    min_val = np.amin([fom])
    max_val = np.amax([fom])

    error = abs(fom-rom)/(max_val-min_val)

    min_error = np.amin([error])
    max_error = np.amax([error])

    im0 = ax[0].imshow(fom[i,...].T[::-1], vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(im0, ax=ax[0], location='top', fraction=0.046, pad=0.04)
    cbar.ax.set_xlabel('FOM', fontsize=12)

    im1 = ax[1].imshow(rom[i,...].T[::-1], vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(im1, ax=ax[1], location='top', fraction=0.046, pad=0.04)
    cbar.ax.set_xlabel('ROM', fontsize=12)

    im2 = ax[2].imshow(error[i,...].T[::-1]/(max_val - min_val)*100, vmin=min_error, vmax=max_error)
    cbar = fig.colorbar(im2, ax=ax[2], location='top', fraction=0.046, pad=0.04)
    cbar.ax.set_xlabel('error %', fontsize=12)

    ax[0].set_ylabel('NY', fontsize=12)
    ax[0].set_xlabel('NX', fontsize=12)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[1].set_xlabel('NX', fontsize=12)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[2].set_xlabel('NX', fontsize=12)
    ax[2].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    plt.tight_layout()

def anim3(fom, rom, frames=None, fig=None, ax=None, interval=25):
    if frames is None:
        frames = fom.shape[0]

    if fig is None and ax is None:
        fig, ax = plt.subplots(1,3, figsize=(13,5.3))

    min_val = np.amin([fom])
    max_val = np.amax([fom])

    
    min_val = np.amin([fom])
    max_val = np.amax([fom])

    error = abs(fom-rom)/(max_val-min_val)

    min_error = np.amin([error])
    max_error = np.amax([error])

    i = 0
    im0 = ax[0].imshow(fom[i,...].T[::-1], vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(im0, ax=ax[0], location='top', fraction=0.046, pad=0.04)
    cbar.ax.set_xlabel('FOM', fontsize=12)

    im1 = ax[1].imshow(rom[i,...].T[::-1], vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(im1, ax=ax[1], location='top', fraction=0.046, pad=0.04)
    cbar.ax.set_xlabel('ROM', fontsize=12)

    im2 = ax[2].imshow(error[i,...].T[::-1]/(max_val - min_val)*100, vmin=min_error, vmax=max_error)
    cbar = fig.colorbar(im2, ax=ax[2], location='top', fraction=0.046, pad=0.04)
    cbar.ax.set_xlabel('error %', fontsize=12)

    def animate(i):
        im0.set_array(fom[i,...].T[::-1])
        im1.set_array(rom[i,...].T[::-1])
        im2.set_array(error[i,...].T[::-1])

    ax[0].set_ylabel('NY', fontsize=12)
    ax[0].set_xlabel('NX', fontsize=12)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[1].set_xlabel('NX', fontsize=12)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[2].set_xlabel('NX', fontsize=12)
    ax[2].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.close()
    anim = FuncAnimation(fig, animate, interval=25, frames=frames)
    #HTML(anim.to_html5_video())
    return anim

def plot_latent( data: list = [], 
                 data_kwargs: list = [], 
                 start_idx = None, 
                 end_idx = None,
                 fig_kwargs: dict = {}, 
                 ax = None,
                 vlines = None,
                 vlines_kwargs: dict = {},
                 xlabel = 'tsteps',
                 label_ncol = 2,
                 label_position = (-0.02, 1.2),
                 **kwargs):
    
    _fig_kwargs = fig_kwargs.copy()

    if 'ncols' not in _fig_kwargs.keys():
        _fig_kwargs['ncols'] = 2
        
    n_columns = _fig_kwargs['ncols']
    n_latent = data[0][1].shape[1]
    if n_latent < n_columns:
        n_columns = n_latent
    n_rows = int(np.ceil(n_latent/n_columns))


    if 'nrows' not in _fig_kwargs.keys():
        _fig_kwargs['nrows'] = n_rows
    if 'figsize' not in _fig_kwargs.keys():
        _fig_kwargs['figsize'] = (7.5*n_columns, n_rows*2)
    if 'colsize' in kwargs.keys():
        _fig_kwargs['figsize'] = (kwargs['colsize'], _fig_kwargs['figsize'][1])
    if 'rowsize' in kwargs.keys():
        _fig_kwargs['figsize'] = (_fig_kwargs['figsize'][0], kwargs['rowsize'])

    #if ax is None:
    fig, ax = plt.subplots( **_fig_kwargs )

    try:
        shape_len = len(ax.shape)
        if shape_len == 1:
            ax = np.array([ax])
    except:
        ax = np.array([[ax]])
        
    vv = 0
    for i in  np.arange(n_rows):
        for j in np.arange(n_columns):
            if vv < n_latent:
                for d,l in zip(data, data_kwargs):
                    time = d[0][start_idx:end_idx]
                    variable = d[1][start_idx:end_idx,vv]
                    ax[i][j].plot(time, variable, **l)
                ax[i][j].set_title(rf" Variable $x_{str({vv+1})}$")
                ax[i][j].grid(True)
                if i == (n_rows-1):
                    ax[i][j].set_xlabel(xlabel)
                if vlines is not None:
                    for vline, vline_kwargs in zip(vlines, vlines_kwargs):
                        ax[i][j].vlines(vline, *ax[i][j].get_ylim(), **vline_kwargs )
            else:
                ax[i][j].axis('off')            
            vv += 1

    #fig.legend(bbox_to_anchor=(0, -0.04))
    ax[0][0].legend(loc='lower left', bbox_to_anchor=label_position, fancybox=True, shadow=True, ncol=label_ncol)
    #plt.legend(bbox_to_anchor=(0, -0.04), fancybox=True, shadow=True, borderaxespad=0)
    plt.tight_layout()
    #plt.show()
    return fig, ax