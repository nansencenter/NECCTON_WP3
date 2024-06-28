#!/bin/bash

module load NCO/5.1.9-iomkl-2022a

infile="daily_climatology_2007_2016.nc"

ncatted -a variable_name,temp,c,c,Temperature $infile
ncatted -a units,temp,c,c,degrees_celsius $infile

ncatted -a variable_name,salin,c,c,Salinity $infile
ncatted -a units,salin,c,c,psu $infile

ncatted -a variable_name,ECO_diac,c,c,Large_phytoplankton_chl-a $infile
ncatted -a units,ECO_diac,c,c,mgChl_m-3 $infile

ncatted -a variable_name,ECO_flac,c,c,Small_phytoplankton_chl-a $infile
ncatted -a units,ECO_flac,c,c,mgChl_m-3 $infile

ncatted -a variable_name,ECO_cclc,c,c,Coccolithophores_chl-a $infile
ncatted -a units,ECO_cclc,c,c,mgChl_m-3 $infile

ncatted -a variable_name,light_sw,c,c,Downwelling_shortwave_flux $infile
ncatted -a units,light_sw,c,c,W_m-2 $infile

ncatted -a variable_name,light_pa,c,c,Photosynthesis_available_radiation $infile
ncatted -a units,light_pa,c,c,W_m-2 $infile

ncatted -a variable_name,mix_dpth,c,c,Mix_layer_depth $infile
ncatted -a units,mix_dpth,c,c,m $infile

ncatted -a variable_name,covice,c,c,Sea-ice_coverage $infile
ncatted -a units,covice,c,c,- $infile

ncatted -a variable_name,thkice,c,c,Sea-ice_thickness $infile
ncatted -a units,thkice,c,c,m $infile
