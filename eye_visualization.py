import numpy as np
import os
import scipy.io as sio
from mayavi import mlab


data = sio.loadmat('./datasets/2015_BOE_Chiu/Subject_05.mat')['images']
data = data.astype(np.uint8)
data.shape = (496,768,61)

mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))
src = mlab.pipeline.scalar_field(data)

blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
voi.trait_set(x_min=0, x_max=193, y_min=0, y_max=125, z_min=0, z_max=60)

mlab.pipeline.iso_surface(voi, contours=[10,200], colormap='Spectral')

mlab.show()