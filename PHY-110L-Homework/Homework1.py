import numpy as np
import matplotlib.pyplot as plt

coords = np.mgrid[-25:26, -25:26]
ygrid = coords[0, :, :]
xgrid = coords[1, :, :]

print(xgrid.shape)
print(ygrid.shape)