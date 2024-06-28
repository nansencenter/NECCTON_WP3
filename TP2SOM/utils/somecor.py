from pathlib import Path
import random

import yaml
import pickle
import numpy as np
import numpy.ma as ma
import calendar
import inspect
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset
import netCDF4 as nc

# SOM package (https://pypi.org/project/sklearn-som/)

from sklearn_som.som import SOM

# load common I/O utilities

from io_utils import NetCDFHandler

# local utilities

def inspect_array(var,isplot):
    """inspect type and shape of a given array"""
    frame = inspect.currentframe().f_back
    try:
        var_name = None
        for name, value in frame.f_locals.items():
            if value is var:
                var_name = name
                break
        print(f"{var_name}: type={type(var)}, shape={var.shape}")
        if isplot and var.ndim == 2:
            plt.imshow(var)
            plt.title(var_name)
            plt.colorbar()
            plt.show()
    finally:
        del frame

def normalize_array(arr):
    # Calculate the standard deviation of the array
    std = ma.std(arr)
    mean = ma.mean(arr)
    
    # Normalize the array
    normalized_arr = ma.masked_array((arr-mean)/std,mask=arr.mask)
    
    return normalized_arr
        
def extract_month_data(data, month, year):
    # Calculate the start and end day for the specified month
    start_day = sum(calendar.monthrange(year, i)[1] for i in range(1, month))
    end_day = start_day + calendar.monthrange(year, month)[1]

    # Extract data for the specified month (0-based indexing)
    month_data = data[start_day:end_day, :, :]
    
    return month_data

# define tasks

isdebug   = False  # print extra array inspection
isplot    = False  # print 2D map with array inspection
issom_cls = True   # perform SOM classification and save pickle file
issom_map = True  # sample plotter for SOM classes on TP2 domain with Basemap

# main

if __name__ == "__main__":

    #------------------------------------
    # user settings
    #------------------------------------

    # Choose how to setup SOM configurarion
    isyml_save = True  # setup SOM configuration here and save yaml file

    if isyml_save:
       isyml_load = False 
    else:
       isyml_load = True  # load SOM configuration from yaml file
       

    # folder to find input files:
    #      topofile = PATH_INPUT_DIR + 'TP2depth.nc'
    #      varfile  = PATH_INPUT_DIR + 'daily_surface_climatology_2007_2016.nc'
    PATH_INPUT_DIR  = '/YOUR/PATH/NECCTON/TP2SOM/'

    # parent folders to save data and figures
    PATH_SAVE_DIR = '../output/SOM/'
    PATH_FIGS_DIR = '../figs/SOM/'

    if isyml_load:
        # specific SOM config file is needed to be selected
        PATH_SOM_CONFIG = PATH_SAVE_DIR + "4x3/01/SOM_config.yaml"
    
    if isyml_save:
        # trial number

        trial="02"
    
        # Control the SOM random generator with random seeding option.
        # For further information, see https://github.com/rileypsmith/sklearn-som

        rseed=random.randint(1,100)
        
        # SOM topology map

        mdim=4
        ndim=3

        # define subset of SOM input parameters by boolean

        name_config_list  = ['dpth','lats','lonx','lony']
        isuse_config_list = [True  ,True  ,False  ,False  ]
    
        name_variables_list  = ['temp','salin','mix_dpth','light_pa','chla','chlmaxday']
        isuse_variables_list = [True  ,True   ,True      ,True      ,True  ,True     ]

        #------------------------------------
        # save the SOM configuration to yaml file
        #------------------------------------

        data = {
           'trial': trial, 
           'rseed': rseed, 
           'mdim': mdim,
           'ndim': ndim,
           'name_config_list': name_config_list,
           'isuse_config_list': isuse_config_list,
           'name_variables_list': name_variables_list,
           'isuse_variables_list': isuse_variables_list
        }

        topology=str(mdim)+'x'+str(ndim)
        PATH_SAVE_YAML = PATH_SAVE_DIR+topology+"/"+trial+"/"
        yaml_filename = PATH_SAVE_YAML+'SOM_config.yaml'

        directory_path = Path(PATH_SAVE_YAML)

        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directories '{directory_path}' created")
        except OSError as error:
            print(f"Error creating directories '{directory_path}': {error}")
        
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
            print(f"SOM config file: {yaml_filename} saved")

    if isyml_load:
        #------------------------------------
        # load the SOM configuration to yaml file (optional)
        #------------------------------------

        yaml_filename = PATH_SOM_CONFIG
    
        with open(yaml_filename, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
            print(f"SOM config file: {yaml_filename} loaded")

        # recover SOM configuration

        trial = data['trial']
        rseed = data['rseed']
        mdim = data['mdim']
        ndim = data['ndim']
        name_config_list = data['name_config_list']
        isuse_config_list = data['isuse_config_list']    
        name_variables_list = data['name_variables_list']
        isuse_variables_list = data['isuse_variables_list']    

    #------------------------------------
    # number of SOM classes
    #------------------------------------
        
    nclss=mdim*ndim 
        
    #------------------------------------
    # I/O settings
    #------------------------------------

    topology=str(mdim)+'x'+str(ndim)
    PATH_SAVE_DATA = PATH_SAVE_DIR+topology+"/"+trial+"/"
    PATH_LOAD_DATA = PATH_SAVE_DIR+topology+"/"+trial+"/"
    PATH_SAVE_FIGS = PATH_FIGS_DIR+topology+"/"+trial+"/"

    #------------------------------------
    # perform SOM classification
    #------------------------------------
    
    if issom_cls:
        #--------------------------------
        # read hycom (TP2) bathemetry
        #--------------------------------

        topofile = PATH_INPUT_DIR + 'TP2depth.nc'

        with NetCDFHandler(topofile) as ncfile:
            # You can now access the data and metadata in the NetCDF file
            print(ncfile.variables.keys())

            # read coordinate
        
            lons_topo_da = ncfile.variables['longitude'][:,:].data
            lats_topo_da = ncfile.variables['latitude'][:,:].data

            # read variable

            dpth_topo_da = ncfile.variables['depth'][:,:].data
            fill_value = ncfile.variables['depth'].getncattr('_FillValue')

            if isdebug:
                print('FillValue',fill_value)

        if isdebug:
            inspect_array(lons_topo_da,isplot)
            inspect_array(lats_topo_da,isplot)
            inspect_array(dpth_topo_da,isplot)

        #--------------------------------
        # convert longitude to point on circle
        #--------------------------------

        lons_topo_rad_da = np.radians(lons_topo_da) # degree to radian
    
        # Calculate x and y coordinates on the unit circle

        lonx_topo_da = np.cos(lons_topo_rad_da)
        lony_topo_da = np.sin(lons_topo_rad_da)

        if isdebug:
            inspect_array(lonx_topo_da,isplot)
            inspect_array(lony_topo_da,isplot)
        
        #--------------------------------
        # define land mask
        #--------------------------------

        land_mask = np.zeros(dpth_topo_da.shape, dtype=bool)
        land_mask[dpth_topo_da >= fill_value] = True
    
        if isdebug:
            inspect_array(land_mask,isplot)
        
        #--------------------------------
        # update land mask
        #--------------------------------
    
        varfile = PATH_INPUT_DIR + 'daily_surface_climatology_2007_2016.nc'
    
        with NetCDFHandler(varfile) as ncfile:
        
            fill_value = ncfile.variables['temp'].getncattr('_FillValue')
            if isdebug:
                print('FillValue',fill_value)

            temp2D_da = ncfile.variables['temp'][0,:,:,:].data.squeeze()
            temp_mask = np.zeros(temp2D_da.shape, dtype=bool)
            temp_mask[temp2D_da <= fill_value] = True
        
            if isdebug:
                inspect_array(temp_mask,isplot)
                inspect_array(land_mask,isplot)
                print('temp_mask', np.sum(temp_mask))
                print('land_mask', np.sum(land_mask))

            land_mask[temp_mask == 1] = 1
            if isdebug:
                print('land_mask', np.sum(land_mask))

        if isdebug:
            inspect_array(land_mask,isplot)

        #--------------------------------
        # apply land mask to config files
        #--------------------------------

        dpth_topo_mskd_ma = ma.masked_array(dpth_topo_da,mask=land_mask)
        lats_topo_mskd_ma = ma.masked_array(lats_topo_da,mask=land_mask)
        lons_topo_mskd_ma = ma.masked_array(lons_topo_da,mask=land_mask)
        lonx_topo_mskd_ma = ma.masked_array(lonx_topo_da,mask=land_mask)
        lony_topo_mskd_ma = ma.masked_array(lony_topo_da,mask=land_mask)

        if isdebug:
            inspect_array(dpth_topo_mskd_ma,isplot)
            inspect_array(lats_topo_mskd_ma,isplot)
            inspect_array(lonx_topo_mskd_ma,isplot)
            inspect_array(lony_topo_mskd_ma,isplot)

        #--------------------------------
        # normalized bathymetry and latitude
        #--------------------------------

        dpth_topo_mskd_norm_ma = normalize_array(dpth_topo_mskd_ma)
        lats_topo_mskd_norm_ma = normalize_array(lats_topo_mskd_ma)
        lonx_topo_mskd_norm_ma = normalize_array(lonx_topo_mskd_ma)
        lony_topo_mskd_norm_ma = normalize_array(lony_topo_mskd_ma)

        #--------------------------------
        # register arrays to dictionary
        #--------------------------------

        arrays_dict = {}
        arrays_norm_dict = {}

        # masked array
    
        arrays_dict['dpth'] = dpth_topo_mskd_ma 
        arrays_dict['lats'] = lats_topo_mskd_ma 
        arrays_dict['lonx'] = lonx_topo_mskd_ma 
        arrays_dict['lony'] = lony_topo_mskd_ma

        if isdebug:
            print(arrays_dict)
    
        if isdebug:
            inspect_array(arrays_dict['dpth'],isplot)
            inspect_array(arrays_dict['lats'],isplot)
            inspect_array(arrays_dict['lonx'],isplot)
            inspect_array(arrays_dict['lony'],isplot)

        # normalized array
    
        arrays_norm_dict['dpth'] = dpth_topo_mskd_norm_ma 
        arrays_norm_dict['lats'] = lats_topo_mskd_norm_ma 
        arrays_norm_dict['lonx'] = lonx_topo_mskd_norm_ma 
        arrays_norm_dict['lony'] = lony_topo_mskd_norm_ma
        
        if isdebug:
            print(arrays_norm_dict)
    
        if isdebug:
            inspect_array(arrays_norm_dict['dpth'],isplot)
            inspect_array(arrays_norm_dict['lats'],isplot)
            inspect_array(arrays_norm_dict['lonx'],isplot)
            inspect_array(arrays_norm_dict['lony'],isplot)

        #--------------------------------
        # read hycom parameters
        #--------------------------------

        isplot = False

        varfile = PATH_INPUT_DIR + 'daily_surface_climatology_2007_2016.nc'

        # read variables
    
        with NetCDFHandler(varfile) as ncfile:
            # You can now access the data and metadata in the NetCDF file
            print(ncfile.variables.keys())

            # define seasons
        
            year = 2007 # any noleap year
            months_jfm = [1,2,3]
            months_jas = [7,8,9]
                
            # read full set of variables

            for name in name_variables_list:
                print(name)
                if name == 'chlmaxday':
                    vars_tmp =  (ncfile.variables['ECO_diac'][:,:,:].data +
                                ncfile.variables['ECO_flac'][:,:,:].data +
                                ncfile.variables['ECO_cclc'][:,:,:].data)
                    vars_tmp = vars_tmp.squeeze()
                    # adjust the maximum chl-a date by setting the time e.g. between Mar and Oct 
                    chlmax_date = np.argmax(vars_tmp[60:270], axis=0)
                    chlmax_date_mskd_ma = ma.masked_array(chlmax_date,mask=land_mask)
                    #-- normalization
                    chlmax_date_mskd_norm_ma = normalize_array(chlmax_date_mskd_ma)
                    #-- register to the dictionary
                    # masked array
                    arrays_dict['chlmaxday'] = chlmax_date_mskd_ma
                    # normalized array
                    arrays_norm_dict['chlmaxday'] = chlmax_date_mskd_norm_ma
                else: 
                    if name == 'chla':
                        vars_da = (ncfile.variables['ECO_diac'][:,:,:].data +
                                   ncfile.variables['ECO_flac'][:,:,:].data +
                                   ncfile.variables['ECO_cclc'][:,:,:].data)
                    else:
                        vars_da = ncfile.variables[name][:,:,:].data

                    vars_da = vars_da.squeeze()
                    
                    if isdebug:
                        inspect_array(vars_da,isplot)

                    #-- annual mean and seasonal mean

                    vars_ann_da = vars_da.mean(axis=0)
                    vars_ann_mskd_ma = ma.masked_array(vars_ann_da,mask=land_mask)

                    if isdebug:
                        inspect_array(vars_ann_da,isplot)
                        inspect_array(vars_ann_mskd_ma,isplot)
                    
                    #-- winter (JFM) mean
                    
                    vars_list = [extract_month_data(vars_da, month, year) for month in months_jfm]
                    vars_jfm_da = np.concatenate(vars_list,axis=0)
                    vars_jfm_da = vars_jfm_da.mean(axis=0)

                    #-- summer (JAS) mean

                    vars_list = [extract_month_data(vars_da, month, year) for month in months_jas]
                    vars_jas_da = np.concatenate(vars_list,axis=0)
                    vars_jas_da = vars_jas_da.mean(axis=0)
                    
                    if isdebug:
                        inspect_array(vars_jfm_da,isplot)
                        inspect_array(vars_jas_da,isplot)
                    
                    #-- delta: seasonal contrast (summer - winter)
                    
                    vars_dlt_da = vars_jas_da - vars_jfm_da
                    vars_dlt_mskd_ma = ma.masked_array(vars_dlt_da,mask=land_mask)

                    if isdebug:
                        inspect_array(vars_dlt_da,isplot)
                        inspect_array(vars_dlt_mskd_ma,isplot)
                    
                    #-- normalization

                    vars_ann_mskd_norm_ma = normalize_array(vars_ann_mskd_ma)
                    vars_dlt_mskd_norm_ma = normalize_array(vars_dlt_mskd_ma)
            
                    if isdebug:
                        inspect_array(vars_ann_mskd_norm_ma,isplot)
                        inspect_array(vars_dlt_mskd_norm_ma,isplot)
                    
                    #-- register to the dictionary
                    
                    # masked array
            
                    arrays_dict['ann_'+name] = vars_ann_mskd_ma
                    arrays_dict['dlt_'+name] = vars_dlt_mskd_ma

                    # normalized array
                    
                    arrays_norm_dict['ann_'+name] = vars_ann_mskd_norm_ma
                    arrays_norm_dict['dlt_'+name] = vars_dlt_mskd_norm_ma
        if isdebug:
            print(arrays_dict)
            print(arrays_norm_dict)
      
        #--------------------------------
        # setup parameters for SOM
        #--------------------------------

        # concatenate numpy.ndarray in 1D array after removing land

        alist = []

        for name, isuse in zip(name_config_list, isuse_config_list):
            if isuse:
                print(name)
                if isdebug:
                    inspect_array(arrays_norm_dict[name],isplot)
                alist.append(arrays_norm_dict[name].compressed()) # compress 2D > 1D (remove land)
                
        for name, isuse in zip(name_variables_list, isuse_variables_list):
            if isuse:
                if name != 'chlmaxday':
                    print('ann_'+name)
                    if isdebug:
                        inspect_array(arrays_norm_dict['ann_'+name],isplot)
                    alist.append(arrays_norm_dict['ann_'+name].compressed())  # compress 2D > 1D (remove land)
                else:
                    print(name)
                    alist.append(arrays_norm_dict[name].compressed())  # compress 2D > 1D (remove land)

        for name, isuse in zip(name_variables_list, isuse_variables_list):
            if isuse:
                if name != 'chlmaxday':
                    print('dlt_'+name)
                    if isdebug:
                        inspect_array(arrays_norm_dict['dlt_'+name],isplot)
                    alist.append(arrays_norm_dict['dlt_'+name].compressed())  # compress 2D > 1D (remove land)
                
        mvars = np.stack(alist,axis=1)
        print(f"mvar shape: {mvars.shape}")
        inspect_array(mvars,isplot)

        isplot = True

        #---------------------------------------
        # Random sampling of mvars along axis=0
        #---------------------------------------

        # compress coordinates
    
        lats_topo_mskd_da_1D = lats_topo_mskd_ma.compressed()
        lons_topo_mskd_da_1D = lons_topo_mskd_ma.compressed()

        # random sampling of input data for training
        
        frac = 1.0 # fraction of mvars to be sampled [0-1]

        dlen = mvars.shape[0]
        random.seed(10)
        inds = random.sample( list(range(dlen)), int(frac*dlen) )
        print('INDS:',inds[:10])

        data_train = mvars[inds,:]
        lons_train = lons_topo_mskd_da_1D[inds]
        lats_train = lats_topo_mskd_da_1D[inds]

        if isdebug:
            inspect_array(data_train,False)
            inspect_array(lons_train,isplot)
            inspect_array(lats_train,isplot)        
            
        #---------------------------------------
        # prepare data to classify
        #---------------------------------------
        
        data_som = mvars
        lons_som = lons_topo_mskd_da_1D # for mapping
        lats_som = lats_topo_mskd_da_1D # for mapping
        
        #---------------------------------------
        # SOM classification
        #---------------------------------------

        # count number of parameters
        try:
            # Get the index of 'chlmaxday' in name_variables_list
            index_of_chlmaxday = name_variables_list.index('chlmaxday')
            chlmaxday_exists = True
        except ValueError:
            # 'chlmaxday' does not exist in name_variables_list
            index_of_chlmaxday = -1
            chlmaxday_exists = False
        # Calculate pdim based on the existence of 'chlmaxday'
        if chlmaxday_exists and isuse_variables_list[index_of_chlmaxday]:  # 'chlmaxday' exists and isuse
            pdim = sum(isuse_config_list) + 2 * (sum(isuse_variables_list) - 1) + 1
        else:
            pdim = sum(isuse_config_list) + 2 * sum(isuse_variables_list)
        print('Number of Parameters: ', pdim)

        # train SOM classes
        tp2_som = SOM(m=mdim,n=ndim,dim=pdim,random_state=rseed)
        tp2_som.fit(data_train)

        # assign SOM classes to given data
        
        clss_som = tp2_som.predict(data_som) 
        print(f"clss_som shape:{clss_som.shape}")

        #---------------------------------------
        # remap SOM classes to TP2 grid
        #---------------------------------------

        original_shape = lats_topo_mskd_ma.shape
        clss_topo_da = np.empty(original_shape, dtype=lats_topo_mskd_ma.dtype)
        clss_topo_da[~land_mask] = clss_som
        
        clss_topo_ma = np.ma.array(clss_topo_da, mask=land_mask)

        # Create the mask to mask out the southern bounary
        mask_sb = (lats_topo_da < 58.5) & (lons_topo_da < 40) & (lons_topo_da > -70)
        clss_topo_ma = np.ma.masked_where(mask_sb, clss_topo_ma)
        
        inspect_array(clss_topo_ma,isplot)

        if isplot:
            inspect_array(lons_topo_da,isplot)
            inspect_array(lats_topo_da,isplot)

        #---------------------------------------
        # save ecoregion to netcdf file
        #---------------------------------------

        jdim = clss_topo_ma.shape[0]
        idim = clss_topo_ma.shape[1]
        # save ecoregion starting from 1 instead of 0
        clss_topo_ma_addone=clss_topo_ma + 1
        
        # Create a new NetCDF file

        file_path=PATH_SAVE_DATA+'TP2_ecor.nc'
        dataset = Dataset(file_path, 'w', format='NETCDF4_CLASSIC')

        # Create dimensions
        dataset.createDimension('jdim',jdim)
        dataset.createDimension('idim',idim)

        # Create variables
        latitudes = dataset.createVariable('latitude', np.float32, ('jdim','idim'))
        longitudes = dataset.createVariable('longitude', np.float32, ('jdim','idim'))
        depths = dataset.createVariable('depth', np.float32, ('jdim','idim'), fill_value=1e+20)
        variables = dataset.createVariable('ecoregion', np.float32, ('jdim','idim'), fill_value=1e+20)

        # Assign attributes
        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'
        depths.units = 'm'
        variables.units = 'None'

        # Write data to variables
        latitudes[:,:] = lats_topo_da
        longitudes[:,:] = lons_topo_da
        depths[:,:] = dpth_topo_mskd_ma
        variables[:,:] = clss_topo_ma_addone.filled(1e+20)  # Use .filled() method to replace masked values

        # Close the dataset
        dataset.close()

        #---------------------------------------
        # save data for analysis
        #---------------------------------------

        directory_path = Path(PATH_SAVE_DATA)

        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directories '{directory_path}' created")
        except OSError as error:
            print(f"Error creating directories '{directory_path}': {error}")

        # dictionary for 2D plot
        
        som_dict = {}
        som_dict['clss_som'] = clss_som
        som_dict['lons_som'] = lons_som
        som_dict['lats_som'] = lats_som
        som_dict['dpth_tp2'] = dpth_topo_da
        som_dict['clss_tp2'] = clss_topo_da
        som_dict['lons_tp2'] = lons_topo_da
        som_dict['lats_tp2'] = lats_topo_da
        som_dict['lmsk_tp2'] = land_mask

        # save som dictionary to pickle file

        file_path = PATH_SAVE_DATA+"som_dict.pkl"
    
        with open(file_path, 'wb') as file:
            pickle.dump(som_dict, file)
            print(f"SOM pickle file: '{file_path}' saved")
    
    #------------------------------------
    # plot ecoregions
    #------------------------------------
    
    if issom_map:
        
        directory_path = Path(PATH_SAVE_FIGS)

        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directories '{directory_path}' created")
        except OSError as error:
            print(f"Error creating directories '{directory_path}': {error}")

        outfile = f"{PATH_SAVE_FIGS}som_tp2_2007-2016_{mdim}x{ndim}_{trial}.png"
        print(outfile) 
    
        #---------------------------------------
        # map SOM classes on TP2 domain with Basemap (add extra colors if nclss > 10)
        #---------------------------------------
        
        from mpl_toolkits.basemap import Basemap, cm
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as cmx
        import pylab as pl

        fig, ax = plt.subplots()
        m = Basemap(projection='npaeqd',resolution='c',boundinglat=55,lon_0=0)
 
        color_list=['tab:orange','tab:blue','tab:red','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','mediumblue','darkcyan','darksalmon','khaki','thistle']
        
        # load som dictionary from pickle file

        file_path = PATH_LOAD_DATA+"som_dict.pkl"
    
        with open(file_path, 'rb') as file:
            som_dict = pickle.load(file)
            print(f"SOM pickle file: '{file_path}' loaded")
    
        clss_som = som_dict['clss_som']
        lons_som = som_dict['lons_som']
        lats_som = som_dict['lats_som']
         
        # Create the mask to mask out the southern bounary
        mask_sb = (lats_som < 58.5) & (lons_som < 40) & (lons_som > -70)
        clss_som = np.ma.masked_where(mask_sb, clss_som)
 
        # plot ecoregion on TP2 domain

        color_dict = {i: color for i, color in enumerate(color_list[:nclss])}
        cmap_var = ListedColormap([color_dict[x] for x in color_dict.keys()])
        x_plot, y_plot = m(lons_som,lats_som)
        cs = m.scatter(x_plot,y_plot,s=4,c=clss_som,edgecolors='none',marker='o',alpha=1.0,cmap=cmap_var,vmin=-0.5, vmax=nclss-0.5)
        cb = m.colorbar(cs,location='right',pad='13%')
        cb.set_ticks([i for i in range(nclss)])
        cb.set_ticklabels([str(i+1) for i in range(nclss)])

        # draw coastlines, country boundaries, fill continents.

        m.drawcoastlines(linewidth=0.5,color='dimgray')
        m.fillcontinents(color='whitesmoke',lake_color='whitesmoke')

        # draw meridians and parallelss

        lbs_lon=[1, 0, 0, 1]
        lbs_lat=[0, 1, 1, 0]

        m.drawmeridians(range(-180,180,20),labels=lbs_lon,color='gray');
        m.drawparallels(range(-90,90,10),labels=lbs_lat,color='gray');

        # title

        title='TP2 som classes: 2007-2016 (trial '+trial+')'
        plt.title(title,y=1.06)

        # save figure

        pl.savefig(outfile,dpi=300)
        plt.show()
        plt.clf()
