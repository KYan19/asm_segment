#!/usr/bin/env python
import os
from pathlib import Path
import tarfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask

data_path = "/n/home07/kayan/asm/data/karena.tar.gz" # home directory
out_path = "/n/holyscratch01/tambe_lab/kayan/" # global scratch

# comment next two lines out if images have already been extracted
#with tarfile.open(data_path, 'r:gz') as tar:
#    tar.extractall(out_path, filter="data")
    
out_path = "/n/holyscratch01/tambe_lab/kayan/karena/"
    
# generates a list of unique image identifiers (strips away file ending)
img_ids = [Path(x).stem for x in os.listdir(out_path+"images") if x.lower().endswith(".tif")]

# geodataframe with mine information
label_df = gpd.read_file(out_path+"/filtered_labels.geojson")

def rasterize(geo,out_shape,transform):
    """Given a geometry, converts to a boolean raster with the geometry marked as True"""
    # if no polygons, return an all-False array
    if geo.isnull().item():
        return np.full(out_shape, False)
        
    # convert polygons to boolean rasters, with mines as True
    else:
        return geometry_mask(
            geo,
            transform=transform,
            invert=True, # so that polygons are marked as True
            out_shape=out_shape,
            all_touched=True
        )

# for each image, save the corresponding mine raster as a .tif file
for img_id in img_ids:
    img = rasterio.open(out_path+"images/"+img_id+".tif")
    rgb_img = img.read([3,2,1]) # RGB channels
    
    # convert GDF to match crs of satellite image
    geo = label_df[label_df["unique_id"] == img_id]["geometry"]
    geo = geo.to_crs(img.crs)
    
    out_shape = rgb_img.shape[1:] # first channel is for RGB; second and third are image dimensions
    transform = img.transform # unique to each image
    
    # create raster
    rasterized_polygon = rasterize(geo, out_shape, transform)
    
    raster_dir = out_path+"rasters/"
    if not os.path.exists(raster_dir):
        os.makedirs(raster_dir)
    raster_path = raster_dir+img_id+".tif"
    
    # save raster
    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=out_shape[0],
        width=out_shape[1],
        count=1,
        dtype='uint8',
        crs=img.crs,
        transform=transform,
    ) as dst:
        dst.write(rasterized_polygon, 1)