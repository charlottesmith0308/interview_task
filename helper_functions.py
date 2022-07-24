#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:54:40 2022

@author: Charlotte Smith

@description: Sylvera Geospatial Data Engineer Technical Task 
"""
# %%
# =============================================================================
# Set Up
# =============================================================================

import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.mask
from rasterio.plot import reshape_as_image, show, adjust_band
import seaborn as sns


# Allow division by zero
np.seterr(divide='ignore', invalid='ignore')



def clip_raster_to_shape(src_path, shape_path, dst_base):
    """
    Clip raster to shape file

    Parameters
    ----------
    src_path : string
        Path to tif file containing raster to be clipped
    shape_path : string
        Path to shape file to be used for clipping

    Returns
    -------
    None.

    """
    
    # Build destination path
    dst_filename = os.path.basename(src_path).replace(".", "_clipped.")
    dst_path = f"{dst_base}{dst_filename}"
    
    # Read shape
    gdf = gpd.read_file(shape_path)
    shape = gdf['geometry'].to_list()
    
    # Open raster
    with rio.open(src_path) as src:
        
        # Clip raster
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)
       
        # Update original meta data to match clipped raster
        out_meta = src.profile   
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        
        # Write clipped raster to file
        with rio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)
            dst.close()

        
def create_true_colour_image(src_path, dst_base):
    """
    Convert 7 band landsat raster into 3 band RGB raster and plot true color 
    image.

    Parameters
    ----------
    src_path : string
        Path to tif file containing 7 band raster to be converted to RGB

    Returns
    -------
    None.

    """
    # Build destination path
    dst_filename = os.path.basename(src_path).replace("_", "_true_color_")
    dst_path = f"{dst_base}{dst_filename}"

    
    # Open raster
    with rio.open(src_path) as src:
        
        # Read RGB bands (B4, B3, B2)
        out_im = src.read((3, 2, 1))
        
        # Update meta data for 3 band raster
        out_meta = src.profile
        out_meta.update({'count':'3'})

        # Write true color raster to file
        with rio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_im)
            dst.close()

        
def calculate_ndvi(src_path):
    """
    Calculate pixel NDVI from composite landsat raster 

    Parameters
    ----------
    src_path : string
        Path to raster to calculate NDVI from.
        Band 3 must be landsat B4
        Band 4 must be landsat B5

    Returns
    -------
    ndvi : np.array
        An arrany of pixel nvdi

    """
    
    # Get RED and NIR bands from source raster
    with rio.open(src_path) as src:
        RED = src.read(3)
        NIR = src.read(4)

    # Calculate NDVI using standard formula
    ndvi = (NIR.astype(float) - RED.astype(float)) / (NIR + RED)
    
    return ndvi


def build_ndvi_time_series(src_files, dst_base):
    """
    Create a time series of the average NDVI for list of rasters and saves a
    to a csv.

    Parameters
    ----------
    src_files : list
        List of the file paths for rasters to be included in the time series. 
        File names must be YYYY_*.tif

    Returns
    -------
    None.

    """
    
    # Build empty dataframe for time series data
    ts = pd.DataFrame(columns=['year', 'ndvi_mean'])
    
    # Populate dataframe for each year
    for f in src_files:
        year = os.path.basename(f)[:4]   # Get raster year
        
        ndvi = calculate_ndvi(f)   # Calculate NDVI
        mean_ndvi = np.mean(ndvi[~np.isnan(ndvi)])  # Mean NDVI excluding NAN 
        
        # Add row to dataframe
        ts = ts.append({'year':year, "ndvi_mean":mean_ndvi}, ignore_index=True)
        ts = ts.sort_values(by='year', ignore_index=True) #Sort chronologically
   
    # Save time series to csv 
    dst_path = f"{dst_base}ndvi_mean_time_series.csv"
    ts.to_csv(dst_path)

def build_RED_time_series_raster(src_files, dst_base):
    """
    Combine the red band from multilpe rasters as a time series stored in a
    single raster.

    Parameters
    ----------
    src_files : list
        A list of file paths for the rasters to be included in the time series

    Returns
    -------
    None.

    """
    # Sort source files chronologically
    src_files.sort()
    
    # Build empty raster for time series
    shape = rio.open(src_files[0]).shape
    ts = np.zeros((len(src_files), shape[0], shape[1]))
    
    # Add RED channel for each year to time series
    for i in range(0, len(src_files)):
        ts[i] = rio.open(src_files[i]).read(3)

    # Create meta data for new time series
    out_meta = rio.open(src_files[0]).profile
    out_meta.update({'count':len(src_files)})
    
    # Build destination path
    dst_path = f"{dst_base}RED_timeseries.tif"
    
    # Write time series to file
    with rio.open(dst_path, "w", **out_meta) as dst:
        dst.write(ts)
        dst.close()

        
def reproject_raster(src_path, dst_base):
    """
    Reproject a raster from WGS 84 to UTM 37S

    Parameters
    ----------
    src_path : string
        File path of raster to be reprojected

    Returns
    -------
    None.

    """
    dst_crs = 'EPSG:32737'   # UTM Zone 37S
    dst_filename = os.path.basename(src_path).replace('.', '_reprojected.')
    dst_path = f"{dst_base}{dst_filename}"
    
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(src.crs, 
                                                               dst_crs, 
                                                               src.width, 
                                                               src.height, 
                                                               *src.bounds
                                                               )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            
            dst.close()
                
                
