#!/bin/bash

# Loop through the years 2008 to 2016
for year in {2008..2016}
do
  # Run the command with the current year
  bash daily2nc.sh $year run
done

