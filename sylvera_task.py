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


# Set working directory
os.chdir('/Users/charlotte/Desktop/Sylvera_Task')


# %%
# =============================================================================
# Task 1 - Clip rasters
# =============================================================================

def clip_raster_to_shape(src_path, shape_path):
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
    dst_path = f"./data/{dst_filename}"
    
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
        src.close()
        
    # Write clipped raster to file
    with rio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_image)
        dst.close()
    

# Clip all tif files in data folder using the provided shape
files_to_clip = glob.glob("./data/*.tif")
shape_path = "./data/AOI/AOI.shp"

for f in files_to_clip:
    clip_raster_to_shape(f, shape_path)



# %%
# =============================================================================
# Task 2 - Plot and save true color image for 2020
# =============================================================================

# Create RGB raster

def create_true_colour_image(src_path):
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

    # Open raster
    with rio.open(src_path) as src:
        
        # Read RGB bands (B4, B3, B2)
        out_im = src.read((3, 2, 1))
        
        # Update meta data for 3 band raster
        out_meta = src.profile
        out_meta.update({'count':'3'})
        
        src.close

    # Build destination path
    dst_filename = os.path.basename(src_path).replace("_", "_true_color_")
    dst_path = f"./data/{dst_filename}"

    # Write true color raster to file
    with rio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_im)
        dst.close()



create_true_colour_image("./data/2020_clipped.tif")
        
        
        
        
# Plot RGB raster
im = rio.open("./data/2020_true_color_clipped.tif").read()
rgb_im= adjust_band(im)
im = reshape_as_image(im)

transform = rio.open("./data/2020_true_color_clipped.tif").transform

fig, ax = plt.subplots(1)
show(rgb_im, ax=ax, transform=transform)

fig.savefig("./data/2020_true_color_clipped.png")


# %%
# =============================================================================
# Task 3 - Create and visualise NDVI time series
# =============================================================================

# Create time series data

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
    
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI using standard formula
    ndvi = (NIR.astype(float) - RED.astype(float)) / (NIR + RED)
    
    return ndvi


def build_ndvi_time_series(src_files):
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
    dst_path = "./data/ndvi_mean_time_series.csv"
    ts.to_csv(dst_path)



src_files = glob.glob("./data/????_clipped.tif")
build_ndvi_time_series(src_files)



# Plot time series data

data = pd.read_csv("./data/ndvi_mean_time_series.csv", index_col=0)

plot = sns.lineplot(data = data,
                    x = 'year',
                    y = 'ndvi_mean',
                    )

plot.set_xlabel('Year')
plot.set_ylabel('Mean NDVI')
sns.despine()



# %%
# =============================================================================
# Task 4 - Save NDVI time series CSV to AWS
# =============================================================================

# Set up session
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# Read dataframe
ndvi_df = pd.read_csv("./data/ndvi_mean_time_series.csv", index_col=0)
key = "files/ndvi_time_series.csv"

# Save to S3
ndvi_df.to_csv(
    f"s3://{AWS_S3_BUCKET}/{key}",
    index=False,
    storage_options={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY,
        "token": AWS_SESSION_TOKEN,
    },
)


# %%
# =============================================================================
# Task 5 - Create time series raster for RED Channel
# =============================================================================


def build_RED_time_series_raster(src_files):
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
        print(i)
        ts[i] = rio.open(src_files[i]).read(3)

    # Create meta data for new time series
    out_meta = rio.open(src_files[0]).profile
    out_meta.update({'count':len(src_files)})
    
    # Build destination path
    dst_path = "./data/RED_timeseries.tif"
    
    # Write time series to file
    with rio.open(dst_path, "w", **out_meta) as dst:
        dst.write(ts)
        dst.close()

        
    
src_files = glob.glob("./data/????_clipped.tif")
build_RED_time_series_raster(src_files)   


# %%
# =============================================================================
# Task 6 - Calculate total area of clipped raster
# =============================================================================

def reproject_raster(src_path):
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
    dst_path = f"./data/{dst_filename}"
    
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



reproject_raster("./data/2020_clipped.tif")


with rio.open("./data/2020_clipped_reprojected.tif") as src:
    im = src.read(1)
    
    px_area = src.res[0] * src.res[1]
    px_count = len(im[im>0])
    
    area = px_area * px_count 
    area_km2 = area / 1000000
    
    print(f'The clipped raster is {np.round(area_km2, 2)} km2')
    



















