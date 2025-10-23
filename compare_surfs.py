import os

import netCDF4 as nc
import numpy as np

from pathlib import Path
from neoplot import plots, figures
from datarie import templates, grids, CLM5


# File paths and file names
outdir = 'out/pft_comparison/'

dir_DETECT: Path = Path('/p/scratch/cjibg31/jibg3105/CESMDataRoot/InputData/lnd/clm2/surfdata_map/EUR_0275/')
dir_NCAR: Path = Path('/p/scratch/cjibg31/jibg3105/CESMDataRoot/InputData/lnd/clm2/surfdata_map/')

file_DETECT: Path = Path('surfdata_EUR-0275_hist_78pfts_Irrig_CMIP6_simyr2000_c230216_GLC2000.nc')
file_NCAR: Path = Path('surfdata_CLM5EU3_v4_pos.nc')



if __name__ == '__main__':

    # create output directory if it does not exist
    os.makedirs(outdir, 
                exist_ok = True)

    # Load the datasets
    ds_DETECT = nc.Dataset(f'{dir_DETECT}/{file_DETECT}', 'r')
    ds_NCAR = nc.Dataset(f'{dir_NCAR}/{file_NCAR}', 'r')

    # Extract PFT variable
    pft_DETECT = ds_DETECT.variables['PCT_NAT_PFT'][:]
    pft_NCAR = ds_NCAR.variables['PCT_NAT_PFT'][:]

    crop_DETECT = ds_DETECT.variables['PCT_CROP'][:]
    crop_NCAR = ds_NCAR.variables['PCT_CROP'][:]

    # Extract the coordinates
    lat = ds_NCAR.variables['LATIXY'][:]
    lon = ds_NCAR.variables['LONGXY'][:]

    # dimensions
    shape = pft_NCAR.shape
    n_pfts = shape[0]

    # output array for differences
    out = np.zeros(shape)

    # grid information for plotting
    EU3_grid = templates.grid(**grids.EU3)

    # Plot the crop area ratio differences
    fig = figures.single_001(projection = 'EU3').create()

    ax = plots.amap(ax = fig.axs[0],
                    lon_extents = EU3_grid.lon_extents,
                    lat_extents = EU3_grid.lat_extents,
                    fs_ticks = 10,
                    fs_label = 10,
                    title = 'PCT_CROP').create()
    
    artist = ax.colormesh(lon = lon,
                          lat = lat,
                          array = crop_DETECT - crop_NCAR,
                          cmap = 'coolwarm_r',
                          cmap_n = 8,
                          vmin = -50,
                          vmax = 50)
    
    colorbar = ax.colorbar(artist = artist,
                           ax = fig.axs,
                           extend = 'both',
                           aspect = 12,
                           fs_label = 10,
                           label = f'difference [%]',
                           shrink = 0.8)
     
    fig.save(Path(f'{outdir}/diff_CROP'))

    # Compare the PFT variables
    # loop for each variable, save diff
    for ip in range(n_pfts):

        # get the "name" of the PFT index
        name = CLM5.CLM5_PFT.names[ip]

        # calculate the difference
        diff = pft_DETECT[ip, :, :] - pft_NCAR[ip, :, :]

        # save the difference to plottable array
        out[ip, :, :] = np.where(diff != 0,
                                 diff,
                                 np.nan)


        # plotting with neoplot
        fig = figures.single_001(projection = 'EU3').create()

        ax = plots.amap(ax = fig.axs[0],
                        lon_extents = EU3_grid.lon_extents,
                        lat_extents = EU3_grid.lat_extents,
                        fs_ticks = 10,
                        fs_label = 10,
                        title = name).create()

        artist = ax.colormesh(lon = lon,
                              lat = lat,
                              array = out[ip, :, :],
                              cmap = 'coolwarm_r',
                              cmap_n = 8,
                              vmin = -100,
                              vmax = 100,)

        colorbar = ax.colorbar(artist = artist,
                               ax = fig.axs,
                               extend = 'both',
                               aspect = 12,
                               fs_label = 10,
                               label = f'difference [%]',
                               shrink = 0.8) 

        fig.save(Path(f'{outdir}/PFT_{ip}'))



    # Overall figure with all PFTs
    figc = figures.fivexthree(projection = 'EU3',
                              fy = 10).create()

    # Loop through each PFT and create a subplot
    for ip in range(n_pfts):

        name = CLM5.CLM5_PFT.names[ip]

        label_lines = [] 
                
        if ip in [12, 13, 14]: label_lines.append('bottom')
        if ip in [2, 5, 8, 11, 14]: label_lines.append('right')

        ax = plots.amap(ax = figc.axs[ip],
                        lon_extents = EU3_grid.lon_extents,
                        lat_extents = EU3_grid.lat_extents,
                        fs_ticks = 10,
                        fs_label = 10,
                        label_lines = label_lines,
                        title = name).create()

        artist = ax.colormesh(lon = lon,
                              lat = lat,
                              array = out[ip, :, :],
                              cmap = 'coolwarm_r',
                              cmap_n = 8,
                              vmin = -100,
                              vmax = 100,)

    colorbar = ax.colorbar(artist = artist,
                           ax = figc.axs,
                           extend = 'both',
                           aspect = 22,
                           fs_label = 10,
                           label = f'difference [%]',
                           shrink = 0.7)

    figc.save(Path(f'{outdir}/all'))
