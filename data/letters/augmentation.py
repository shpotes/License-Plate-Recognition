import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import itertools
from imgaug import augmenters as iaa

from tqdm import tqdm


N_seq = iaa.Sequential([
    iaa.Fliplr(0.05),
    iaa.Flipud(0.05),
    iaa.Dropout(),
    iaa.PerspectiveTransform(),
    iaa.PiecewiseAffine(),
    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255))
], random_order=True)

DIRR = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
IMG = []
def filterr(x, y):
    if x[0] == '-':
        return y
    else:
        return 255 - y

for i in tqdm(DIRR):
    IMG = os.listdir(i)
    imgs = np.array([filterr(x, cv2.cvtColor(cv2.resize(cv2.imread(i + '/' + x), (28, 28)),
                                          cv2.COLOR_BGR2GRAY)) for x in IMG])
    normal = N_seq.augment_images(imgs)
    for j in range(len(IMG)):
        cv2.imwrite('{}/A_{}.jpg'.format(i, j), normal[j,:,:])
