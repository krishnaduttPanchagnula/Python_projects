import numpy as np
import gdal
import ogr
from sklearn import metrics

lcz_f= 'D:/LCZ/lcz_clipped.tif'

driverTiff = gdal.GetDriverByName('GTiff')
lcz = gdal.Open(naip_fn)

test_fn = 'D:/LCZ/lcz_test.shp'
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', lcz.RasterXSize, lcz.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(lcz.GetGeoTransform())
target_ds.SetProjection(lcz.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open('D:/lcz/LCZ_classified.tif')
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

cm = metrics.confusion_matrix(truth[idx], pred[idx])

# pixel accuracy
print(cm)

print(cm.diagonal())
print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)
