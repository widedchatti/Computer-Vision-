import joblib
import skimage
from skimage import transform,feature
import numpy as np

#Create a window that iterates over patches of this image, and compute HOG features for each patch
def sliding_window(img, patch_size=(62, 47), istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Nj, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

#process the image 
def image_processing(data):
    data = skimage.color.rgb2gray(data)
    data = skimage.transform.rescale(data, 0.5)
    data = data[:180, 40:180]
    return data

#prediction function
def predict(data):
    indices, patches = zip(*sliding_window(data))
    patches_hog = np.array([feature.hog(patch) for patch in patches])
    clf = joblib.load("face_detector.sav")
    return clf.predict(patches_hog),indices
