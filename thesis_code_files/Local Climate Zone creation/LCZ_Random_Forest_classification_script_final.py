import gdal
import ogr

# open different climatic data as  gdal raster dataset
lcz_f = 'D:/lcz/clipped.tif'
lcz = gdal.Open(lcz_f)

# open the points file to use for training data
train_f = 'C:/lcz/lcz/train.shp'
train_ds = ogr.Open(train_f)
lyr = train_ds.GetLayer()

# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', lcz.RasterXSize, lcz.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(lcz.GetGeoTransform())
target_ds.SetProjection(lcz.GetProjection())

# rasterize the training points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

# retrieve the rasterized data and print basic stats
data = target_ds.GetRasterBand(1).ReadAsArray()
print('min', data.min(), 'max', data.max(), 'mean', data.mean())


#Groundtruthing data 

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]
print('class values', classes)

segments_per_class = {}

for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print("Training segments for class", klass, ":", len(segments_of_class))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"

# Preparing data for Random forest Classification
train_img = np.copy(segments)
threshold = train_img.max() + 1

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print('Training objects for class', klass, ':', len(class_train_object))

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(training_objects, training_labels)

predicted = classifier.predict(objects)


clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass



mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0



clfds = driverTiff.Create('D:/temp/naip/classified.tif', naip_ds.RasterXSize, naip_ds.RasterYSize,
                          1, gdal.GDT_Float32)
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print('Bro Chill..We have done it!')


# pixel accuracy
print(cm)

print(cm.diagonal())
print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
print(accuracy)
