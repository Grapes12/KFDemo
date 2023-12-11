import numpy as np
import matplotlib.pyplot as plt
from pandas import *

import ODKF
from ODKF import gaussian



data = read_csv("dat.csv")
 
gx = data['gx'].tolist()
gy = data['gy'].tolist()
gz = data['gz'].tolist()
time = np.divide(np.array(data['t'].tolist()), pow(10, 6))

x, y, z = [0], [0], [0]


pgx, pgy, pgz = [], [], []

predict = gaussian(0., 0)

for n, i in enumerate(gx):

    prior = ODKF.predict(predict, gaussian(0., 0.001**2))   
    predict =  ODKF.update(prior, gaussian(i, 0.014**2))

    pgx.append(predict.mean)

    if(n == 0):
        x.append(x[n] + predict.mean * 10018/pow(10,6))
    else:
        x.append(x[n - 1] + predict.mean *(time[n] - time[n-1]))


for n, i in enumerate(gy):

    prior = ODKF.predict(predict, gaussian(0., 0.001**2))   
    predict =  ODKF.update(prior, gaussian(i, 0.014**2))

    pgy.append(predict.mean)

    if(n == 0):
        y.append(y[n] + predict.mean * 10018/pow(10,6))
    else:
        y.append(y[n - 1] + predict.mean * (time[n] - time[n-1]))

for n, i in enumerate(gz):

    prior = ODKF.predict(predict, gaussian(0., 0.001**2))   
    predict =  ODKF.update(prior, gaussian(i, 0.014**2))

    pgz.append(predict.mean)

    if(n == 0):
        z.append(z[n] + predict.mean * 10018/pow(10,6))
    else:
        z.append(z[n - 1] + predict.mean * (time[n] - time[n-1]))


fig, axs = plt.subplots(3,2)
axs[0,0].plot(time,pgx)
axs[1,0].plot(time,pgy)
axs[2,0].plot(time,pgz)

axs[0,0].set_title("Smoothed gx")
axs[1,0].set_title("Smoothed gy")
axs[2,0].set_title("Smoothed gz")

axs[0,0].scatter(time,gx, color='red')
axs[1,0].scatter(time,gy, color='red')
axs[2,0].scatter(time,gz, color='red')


axs[0,1].plot(time,x[1:])
axs[1,1].plot(time,y[1:])
axs[2,1].plot(time,z[1:])

axs[0,1].set_title("x(deg)")
axs[1,1].set_title("y(deg)")
axs[2,1].set_title("z(deg)")

plt.show()

