#-------------------------------------------------------------------------------
# Name:        Create image tiles of a range of raster maps (aerial images and other data)
# Purpose:
#
# Author:      Maarten van Strien
#
# Created:     
# Copyright:   
#-------------------------------------------------------------------------------
##############################
# # Load functions
##############################

import rasterio
import os
import numpy as np
import pandas as pd
from scipy import stats
##############################
# # Settings
##############################
output_folder = r"D:\DCEC"  # The output folder for the tiles
output_filename = "HighRes_data_tiles_2023"
input_folder = r"D:\DCEC"  # The folder containing the input rasters
Nodata = -32768
# These rasters should have the same origin, resolution, and extent.
# The value -32728 is reserved for the no-data value.
input_raster = "MKInput_30m_f16.tif"

winSize = 4  # Size of the tiles to be extracted from the rasters
stride = 4  # Number of pixels to move the window after each extraction

##############################
# # Check dimensions of input rasters
##############################
with rasterio.open(os.path.join(input_folder, input_raster)) as src:
    height, width = src.height, src.width
    total_bands = src.count  # Total number of bands in the raster
    last_10_bands = range(1, total_bands)  # Indices for the last 10 bands (1-based indexing)

    # Initialize the NumPy array for the last 10 bands
    n_bands = len(last_10_bands)
    np_all_bands = np.zeros((height, width, n_bands), dtype=np.int16)

    # Read and stack each band
    for idx, band_idx in enumerate(last_10_bands):
        np_all_bands[:, :, idx] = src.read(band_idx, masked=True).filled(fill_value = Nodata)

# Assert the final shape
assert np_all_bands.shape[2] == n_bands, f"Expected {n_bands} bands, but got {np_all_bands.shape[2]}"

# Print the dimensions of the stacked array
print(f"Stacked array shape: {np_all_bands.shape} (Height, Width, Bands)")

##############################
# # Create tiles
##############################
# Create steps for the sliding window
steps_col = np.arange(0, np_all_bands.shape[1] - winSize + 1, stride)
steps_row = np.arange(0, np_all_bands.shape[0] - winSize + 1, stride)
n_tiles = len(steps_col) * len(steps_row)  # Total number of tiles

# Initialize the numpy array to store all tiles
np_image_stack = np.zeros((n_tiles, winSize, winSize, n_bands), dtype=np.int16)
tile_counter = 0  # Tile index counter
ij_combs = []  # To track the indices of included tiles

# Loop through rows and columns to extract tiles
for i in steps_row:
    for j in steps_col:
        tile_array = np_all_bands[i:i + winSize, j:j + winSize, :]
        if not np.any(tile_array == -32768):  # Skip tiles with NoData values
            np_image_stack[tile_counter, ...] = tile_array  # Add valid tile to the stack
            tile_counter += 1
            ij_combs.append([i, j])  # Record the i, j index of this tile

# Trim the array to the actual number of valid tiles
np_image_stack = np_image_stack[:tile_counter]

# Convert ij_combs list to DataFrame
ij_included = pd.DataFrame(ij_combs, columns=['i', 'j'])

# register the label
with rasterio.open(os.path.join(input_folder, input_raster)) as label:
    label_data = label.read(11, masked = True)
   
labels = []

for idx, (i, j) in enumerate(ij_combs):
    if i + winSize <= label_data.shape[0] and j + winSize <= label_data.shape[1]:
        tile_label = label_data[i:i + winSize, j:j + winSize]
        print(f"Before conversion: {type(tile_label)}")
        tile_label = np.array(tile_label)
        if np.ma.is_masked(tile_label):
            tile_label = tile_label.filled(fill_value=0)  
        print(f"Tile {idx}: Position ({i}, {j}) - Tile content:\n{tile_label}")
        print(f"After conversion: {type(tile_label)}")
        valid_labels = tile_label[(tile_label >= 1) & (tile_label <= 12)]
        print(f"Tile {idx}: Valid labels found: {valid_labels}")
        if valid_labels.size > 0:
            mode_label = np.bincount(valid_labels).argmax()  # Compute the prevalent label
        else:
            mode_label = 0
        labels.append(mode_label)    
    else:
        labels.append(0)         
print(f"Processed {len(labels)} tiles.")
print(f"First few labels: {labels[:10]}")  #  
# Add the calculated labels to the DataFrame
if len(labels) == len(ij_included): 
    ij_included['label'] = labels
else:
    print(f"Mismatch: ij_included has {len(ij_included)} rows, but labels has {len(labels)} labels.")
##############################
# # Save output
##############################
# Save the image stack to a .npz file
np.savez(os.path.join(output_folder, output_filename), data=np_image_stack)

# Save the ij_included DataFrame to a CSV file
ij_included.to_csv(os.path.join(output_folder, output_filename + "_ij_included.csv"), index=False)
print(f"Process completed. {tile_counter} valid tiles were extracted.")