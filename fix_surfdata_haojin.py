import rasterio
from rasterio.transform import rowcol

import numpy as np
import os
import gzip
import glob
import re

import xarray as xr
from scipy.interpolate import griddata

def load_flt_file(filepath):
    open_func = gzip.open if filepath.endswith(".gz") else open
    with open_func(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    data = data.reshape((ncells, nmonths)).T.reshape((nmonths, nrows, ncols))
    return data

ncols, nrows, nmonths = 4320, 2160, 12
ncells = ncols * nrows
data_dir = "/p/project/cjibg36/jibg3674/monthly_growing_area_grids_additional/"


files = glob.glob(os.path.join(data_dir, "crop_*_*_12.flt*"))
crop_files = {}

for fp in files:
    match = re.search(r'crop_(\d+)_([a-zA-Z]+)_12\.flt(?:\.gz)?', os.path.basename(fp))
    if match:
        crop_id, irrig_type = match.groups()
        crop_files.setdefault(crop_id, {})[irrig_type.lower()] = fp

irrigated_sum = np.zeros((nmonths, nrows, ncols), dtype=np.float32)
rainfed_sum = np.zeros((nmonths, nrows, ncols), dtype=np.float32)

# Only process crops with both irrigated and rainfed data
for crop_id, types in sorted(crop_files.items()):
    if 'irrigated' in types and 'rainfed' in types:
        irrigated_data = load_flt_file(types['irrigated'])
        rainfed_data = load_flt_file(types['rainfed'])
        irrigated_sum += irrigated_data
        rainfed_sum += rainfed_data
    else:
        print(f"Skipping crop {crop_id}: missing irrigated or rainfed data.")

# Create mask: where both are zero
zero_mask = (irrigated_total == 0) & (rainfed_total == 0)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    rainfed_ratio = rainfed_total / (rainfed_total+irrigated_total)*100
rainfed_ratio[zero_mask] = np.nan


file_lsm = '/p/project/cjibg36/jibg3674/shared_DA/EUR-11_TSMP_FZJ-IBG3_444x432_LAND-LAKE-SEA-MASK.nc'

grid_centre = xr.open_dataset(file_lsm, decode_times=False)

lats = grid_centre['lat'].values
lons = grid_centre['lon'].values

lat_min, lat_max = lats.min(), lats.max()
lon_min, lon_max = lons.min(), lons.max()

# Read cell area raster
with rasterio.open('/p/project/cjibg36/jibg3674/cell_area_grid/cell_area_ha_05mn.asc') as src:
    area_grid = src.read(1)
    area_nodata = src.nodata
    area_transform = src.transform

# Get coordinates of the source grid
rows, cols = np.indices(area_grid.shape)
src_xs, src_ys = rasterio.transform.xy(area_transform, rows, cols)
src_xs = np.array(src_xs)
src_ys = np.array(src_ys)

# Convert lat/lon to pixel indices using rasterio’s transform
row_min, col_min = rowcol(area_transform, lon_min, lat_max)  # upper-left
row_max, col_max = rowcol(area_transform, lon_max, lat_min)  # lower-right

# Clip indices to array bounds
row_min = np.clip(row_min, 0, area_grid.shape[0]-1)
row_max = np.clip(row_max, 0, area_grid.shape[0]-1)
col_min = np.clip(col_min, 0, area_grid.shape[1]-1)
col_max = np.clip(col_max, 0, area_grid.shape[1]-1)

# Crop area and rainfed arrays
area_crop = area_grid[row_min:row_max+1, col_min:col_max+1]
rainfed_crop = rainfed_ratio[row_min:row_max+1, col_min:col_max+1]

# Get the cropped coordinate grid
rows, cols = np.indices(area_crop.shape)
crop_transform = area_transform * rasterio.Affine.translation(col_min, row_min)
crop_xs, crop_ys = rasterio.transform.xy(crop_transform, rows, cols)
crop_xs = np.array(crop_xs)
crop_ys = np.array(crop_ys)

# Flatten for interpolation
flat_points = np.column_stack((crop_xs.ravel(), crop_ys.ravel()))
flat_values = rainfed_crop.ravel()

target_points = np.column_stack((lons.ravel(), lats.ravel()))

interpolated = griddata(flat_points, flat_values, target_points, method='nearest')
interpolated = interpolated.reshape(lats.shape)

ref_path = "/p/project/cjibg36/jibg3674/shared_DA/setup_eclm_cordex_444x432_v9/input_clm/surfdata_EUR-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230808_GLC2000.nc"
ref_ds = xr.open_dataset(ref_path)

pct_natveg = ref_ds['PCT_NATVEG'] 
pct_crop = ref_ds['PCT_CROP'] 
nat_pft_frac = ref_ds['PCT_NAT_PFT'] # (natpft, lsmlat, lsmlon)
cft_frac = ref_ds['PCT_CFT'] # (cft, lsmlat, lsmlon)

# --- Modify PCT_NAT_PFT ---
nat_pft_frac_new = ref_ds['PCT_NAT_PFT'].copy()
nat_pft_frac_new[0, :, :] = bare_frac_2

# Rescale PFT 0–14 to sum to 100%
total = nat_pft_frac_new[:15, :, :].sum(dim='natpft')
nat_pft_frac_new[:15, :, :] = (nat_pft_frac_new[:15, :, :] / total) * 100.0

# Initialize CFT1 fraction array with default value of 100% everywhere
cft1 = np.full_like(pct_crop, 100.0)

# Create a mask for locations where crop is present
mask = (pct_crop != 0)

# Create a mask for valid interpolated rainfed_ratio values (i.e., not NaN)
valid_ratio = ~np.isnan(interpolated)

# Combine masks: we only want to update where crop exists and the ratio is valid
combined_mask = mask & valid_ratio

# Assign interpolated rainfed ratio to CFT1 where both conditions are satisfied
cft1[combined_mask] = interpolated[combined_mask]

# Copy original PCT_CFT (crop functional type fractions) from dataset
cft_frac_new = ref_ds['PCT_CFT'].copy()

# Assign updated CFT1 fraction (index 0) and complement to CFT2 (index 1)
cft_frac_new[0, :, :] = cft1            # CFT1: e.g., rainfed
cft_frac_new[1, :, :] = 100.0 - cft1    # CFT2: e.g., irrigated

# --- Create and save to a NetCDF with the updated variables ---
new_ds = ref_ds.copy()
new_ds['PCT_NAT_PFT'] = nat_pft_frac_new
new_ds['PCT_CFT'] = cft_frac_new
new_ds.to_netcdf("/p/project/cjibg36/jibg3674/shared_DA/setup_eclm_cordex_444x432_v9/input_clm/surfdata_EUR-11_hist_16pfts_Irrig_CMIP6_simyr2000_c230808_GLC2000_MIRCA2000.nc")


