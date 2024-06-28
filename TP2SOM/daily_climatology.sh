#!/bin/bash

# Loop over days from 001 to 365
for day in $(seq -w 001 365)
do
  # Apply ncra for each day
  ncra -F -d rdim,$day data/archm.*_12.nc data/temp$day.nc
done

# Concatenate the temporary files
ncrcat -h data/temp???.nc data/daily_climatology_2007_2016.nc

# Remove the temporary files
rm temp???.nc

