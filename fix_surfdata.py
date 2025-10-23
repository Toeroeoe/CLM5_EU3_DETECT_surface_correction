import os
import numpy as np
import netCDF4 as nc
import xarray as xr

from pathlib import Path
from neoplot import plots, figures
from datarie import templates, grids

# Paths and file names
outdir = 'out/pft_correction/'
outfile = 'surfdata_EUR-0275_hist_78pfts_Irrig_CMIP6_simyr2000_c230216_GLC2000_corrected_3.nc'
path_surf: Path = Path('/p/scratch/cjibg31/jibg3105/CESMDataRoot/InputData/lnd/clm2/surfdata_map/EUR_0275/')
file_surf: str = 'surfdata_EUR-0275_hist_78pfts_Irrig_CMIP6_simyr2000_c230216_GLC2000.nc'

path_surf_corr: Path = Path('in/')
file_surf_corr: str = 'surfdata_CLM_EUR-0275_TSMP_FZJ-IBG3_CLMPFLDomain_1592x1544.nc'

# Settings
do_correct = True
plot_total_pft_pct = True


if __name__ == '__main__':

    if do_correct:
        # create output directory if it does not exist
        os.makedirs(outdir, 
                    exist_ok = True)

        # Load surface file data
        data_corr = xr.open_dataset(f'{path_surf_corr}/{file_surf_corr}')
        data_surf = xr.open_dataset(f'{path_surf}/{file_surf}')

        new_surf_array = data_surf['PCT_NAT_PFT'].copy()
        landmask = data_surf['LANDFRAC_PFT'].copy()
        bareground = data_corr['PCT_PFT'][0, :, :].copy()  # Assuming first index is bareground

        bareground = np.where(landmask == 0,
                              100,
                              bareground)

        new_surf_array[0, :, :] = bareground

        ind_zeros = np.argwhere(np.sum(new_surf_array.values, axis=0) == 0)

        for i in range(ind_zeros.shape[0]):
            new_surf_array.loc[dict(natpft=0, 
                                    lsmlat=ind_zeros[i, 0], 
                                    lsmlon=ind_zeros[i, 1])] = 100.0

        new_total = new_surf_array.sum(dim='natpft')

        new_surf_array_scaled = (new_surf_array / new_total) * 100.0

        new_surf_array_scaled_w0 = np.where(np.isnan(new_surf_array_scaled), 
                                            0, 
                                            new_surf_array_scaled)

        data_surf['PCT_NAT_PFT'].data = new_surf_array_scaled_w0

        # Save the modified surface data
        data_surf.to_netcdf(f'{outdir}/{outfile}',
                           mode='w',
                           format='NETCDF4_CLASSIC')

    
    if plot_total_pft_pct:

        # Load grid information for plotting
        EU3_grid = templates.grid(**grids.EU3)

        ds_surf = nc.Dataset(f'{outdir}/{outfile}', 'r')

        # Read the PFT ratio variable
        pct_pft = ds_surf.variables['PCT_NAT_PFT'][:]
        lat = ds_surf.variables['LATIXY'][:]
        lon = ds_surf.variables['LONGXY'][:]

        sum_pct_pft = np.sum(pct_pft, axis = 0)

        sum_pct_pft_unique = np.unique(sum_pct_pft, return_counts=True)

        for value, count in zip(*sum_pct_pft_unique):
            print(f'value: {value}, count: {count}')

        exit()

        if np.any(sum_pct_pft != 100):

            count = np.count_nonzero(sum_pct_pft != 100)
            print(f"Warning: {count} invalid values found in sum_pct_pft.")

            count = np.count_nonzero(sum_pct_pft == 100)
            print(f"Warning: {count} valid values found in sum_pct_pft.")
        
        # Create figure intance
        fig = figures.single_001(projection = 'EU3').create()

        # Create axes instance
        ax = plots.amap(ax = fig.axs[0],
                        lon_extents = EU3_grid.lon_extents,
                        lat_extents = EU3_grid.lat_extents,
                        fs_ticks = 10,
                        fs_label = 10,
                        title = 'PFT PCT total').create()

        # Create the colormesh artist
        artist = ax.colormesh(lon = lon,
                              lat = lat,
                              array = sum_pct_pft,
                              cmap = 'coolwarm_r',
                              cmap_n = 4,
                              vmin = 0,
                              vmax = 1)

        # Add colorbar to the figure
        colorbar = ax.colorbar(artist = artist,
                               ax = fig.axs,
                               extend = 'both',
                               aspect = 12,
                               fs_label = 10,
                               label = f'sum %',
                               shrink = 0.8)

        fig.save(Path(f'{outdir}/PCT_PFT_total_2'))

    
    