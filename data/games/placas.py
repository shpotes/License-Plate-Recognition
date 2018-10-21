import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.chdir('../placas-dataset/caracteres-placas/')
IMAGES = []
for files in os.listdir():
    IMAGES.append((cv2.cvtColor(cv2.imread(files), cv2.COLOR_BGR2GRAY), files[0]))

MIN = min(map(lambda x: min(x[0].shape), IMAGES))
data = [(cv2.resize(x[0], dsize=(MIN, MIN), interpolation=cv2.INTER_CUBIC), x[1]) for x in IMAGES]

#plt.imshow()
#plt.show()
