from osgeo import gdal, ogr,osr
import sys
import os


gdal.UseExceptions()

#Opening the file
fileName = "D:/lcz/LCZ_classified.tif"

src_ds = gdal.Open(fileName)
if src_ds is None:
    print('Unable to open %s' % src_fileName)
    sys.exit(1)
    
#obtainning the sspatial projection from the source file    

srs = osr.SpatialReference()
srs.ImportFromWkt(src_ds.GetProjection())    
    
srcband = src_ds.GetRasterBand(1)
dst_layername = "PolyFtr"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource(dst_layername + ".shp")
dst_layer = dst_ds.CreateLayer(dst_layername, srs = None)
newField = ogr.FieldDefn('Area', ogr.OFTInteger)
dst_layer.CreateField(newField)
gdal.Polygonize(srcband, None, dst_layer, 0, [], 
callback=None )
