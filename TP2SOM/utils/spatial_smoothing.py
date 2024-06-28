# Purpose: apply spatial smoothing of a 2D field of categorical data from a NetCDF file

import numpy as np
import netCDF4 as nc
import os

# Function to perform the majority filter while preserving land mask
# Moving window size can be changed with the size parameter, e.g. size=5 meaning 5x5 pixels
def majority_filter_preserve_land(data, mask, size=5,pad_value=-9999, land_value=1.e+20):
    pad_size = size // 2
    padded_data = np.pad(data, pad_size, mode='constant', constant_values=pad_value)
    padded_mask = np.pad(mask, pad_size, mode='constant', constant_values=True) 
    result = data.copy()

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #print(i,j,mask[i,j])
            if mask[i, j]:  # Skip land pixels
                result[i, j] = land_value
            else:
                # Calculate indices for centered window
                start_i = i - size//2 + pad_size
                end_i = i + size//2 + pad_size + 1
                start_j = j - size//2 + pad_size
                end_j = j + size//2 + pad_size + 1 

                window = padded_data[start_i:end_i, start_j:end_j]
                window_mask = padded_mask[start_i:end_i, start_j:end_j]

                # Filter out land pixels from window
                non_land_pixels = window[window_mask == False]
                
                # Exclude padding values
                valid_pixels = non_land_pixels[non_land_pixels != pad_value]  
                
                # Find the most common value among valid pixels in the window
                values, counts = np.unique(valid_pixels, return_counts=True)
                majority_value = values[np.argmax(counts)]
                result[i, j] = majority_value
    return result

# Function to perform smoothing multiple times while preserving land mask
def apply_smoothing_preserve_land(data, mask, iterations, size=5):
    smoothed_data = data.copy()
    for _ in range(iterations):
        # Apply majority filter while preserving land mask
        smoothed_data = majority_filter_preserve_land(smoothed_data, mask, size)
    return smoothed_data

#------------------------------------
# user settings
#------------------------------------

# Define input path
PATH_SAVE_DIR = '../output/SOM/'
topology='4x3'
trial='82'
PATH_SAVE_DATA = f"{PATH_SAVE_DIR}{topology}/{trial}/"

# Read the NetCDF file which will be smoothed
file_name = PATH_SAVE_DATA + "TP2_ecor.nc"

# Define the number of smoothing iterations
num_iterations = 6

#--------------------------------------
# Extract the variables
ds = nc.Dataset(file_name)
ecoregion = ds.variables['ecoregion'][:]
#land_mask = np.where(ecoregion <= 12 , False, True)
land_mask = np.ma.getmaskarray(ecoregion) 

# Perform the smoothing while preserving the land mask
smoothed_ecoregion = apply_smoothing_preserve_land(ecoregion, land_mask, num_iterations)

# Define the path and name of output file 
output_file = f"{PATH_SAVE_DATA}smoothed_ecoregion_{num_iterations}_iteration.nc"
# Check if the file exists
if os.path.exists(output_file):
    # If the file exists, remove it
    os.remove(output_file)

# Save the final smoothed ecoregion data back to a NetCDF file
with nc.Dataset(output_file, 'w') as output_ds:
    # Copy dimensions
    for dim_name, dim in ds.dimensions.items():
        output_ds.createDimension(dim_name, len(dim))

    # Copy variables except ecoregion
    for var_name, var in ds.variables.items():
        if var_name == 'ecoregion':
            continue
        out_var = output_ds.createVariable(var_name, var.datatype, var.dimensions)
        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out_var[:] = var[:]

    # Create the final smoothed ecoregion variable
    out_ecoregion = output_ds.createVariable('ecoregion', ecoregion.dtype, ('jdim', 'idim'))
    out_ecoregion.setncatts({k: ds.variables['ecoregion'].getncattr(k) for k in ds.variables['ecoregion'].ncattrs()})
    out_ecoregion[:] = smoothed_ecoregion

print(f"Smoothing completed with {num_iterations} iterations and saved to", output_file)

