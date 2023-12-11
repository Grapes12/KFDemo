import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import namedtuple
from pandas import *

import ODKF
from ODKF import gaussian

#gaussian = namedtuple('Gaussian', ['mean', 'var'])
#gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'


data = read_csv("dat.csv")
 
gx = data['gx'].tolist()
gy = data['gy'].tolist()
gz = data['gz'].tolist()
time = data['t'].tolist()

 
fig, axs = plt.subplots(3)

axs[0].plot(gx, gy)
axs[1].plot(gx, gz)
axs[2].plot(gy, gz)

for i in axs:
    i.set_aspect('equal', adjustable='box')

plt.show()

