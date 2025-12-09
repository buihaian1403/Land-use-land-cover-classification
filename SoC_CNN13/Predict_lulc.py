import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

# Input settings
predFile = "Soc_prediction.csv"  # Path to the single prediction CSV file
blueprint_raster = "D:/DCEC/MKInput_5m.tif"
inFolder = "D:/SoC4SS_FGVC/"
outFolder = "D:/SoC4SS_FGVC/"
winSize = 24
stride = 24  # Size of the tiles
NODATA = -128  # Define the NODATA value used for padding

# Load the predictions from file
df = pd.read_csv(os.path.join(inFolder, predFile)) 
positions = df[['i', 'j']].values  # Extract positions (i, j)
yPred = df['label'].values  # Third column is the predicted labels

# Extract properties of the blueprint raster
with rasterio.open(blueprint_raster) as src:
    # Get the dimensions, transform, and CRS (coordinate reference system)
    blueprint_transform = src.transform
    blueprint_crs = src.crs
    columns, rows = src.width, src.height
    cell_size_x = abs(src.transform[0]) * winSize
    cell_size_y = abs(src.transform[4]) * winSize
    Xmin, Ymax = src.bounds.left, src.bounds.top

if winSize != stride:
    raise ValueError("ERROR: winSize is not equal to stride. Use another method to construct the prediction raster.")

# Ensure the number of rows in ijIncl matches the length of yPred
if positions.shape[0] != len(yPred):
    raise ValueError("ERROR: Length of ijIncl and yPred not equal")

# Determine the number of steps in column and row directions
steps_col = len(np.arange(0, columns - winSize + 1, stride))
steps_row = len(np.arange(0, rows - winSize + 1, stride))
n_tiles = steps_col * steps_row  # Total number of tiles

# Initialize an array to store predictions
np_features = np.full((steps_row, steps_col), -128, dtype=np.int8)  # Default to Nodata value (-128)

# Loop through the tiles and add the predicted landscape type
for row in range(positions.shape[0]):
    i = int(positions[row,0] // winSize)  # Get tile index in row direction
    j = int(positions[row,1] // winSize)  # Get tile index in column direction
    pred = yPred[row]  # Get prediction
    np_features[i, j] = pred  # Store in features array

# Define the raster transformation for the output raster
transform = rasterio.transform.from_origin(Xmin, Ymax, cell_size_x, cell_size_y)

# Define the metadata for the output raster
out_meta = {
    "driver": "GTiff",
    "height": np_features.shape[0],
    "width": np_features.shape[1],
    "count": 1,
    "dtype": np_features.dtype,
    "crs": blueprint_crs,
    "transform": transform,
    "nodata": -128
}

# Save the predictions as a GeoTIFF
output_path = os.path.join(outFolder, "Soc_pred.tif")
with rasterio.open(output_path, 'w', **out_meta) as dst:
    dst.write(np_features, 1)

print(f"Prediction raster saved to {output_path}")