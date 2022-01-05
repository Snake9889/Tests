#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy import (linspace, sin, cos, pi)
from numpy.fft import rfft, rfftfreq
from sklearn.decomposition import FastICA

# ===================================================================================

x = linspace(0, 1000, 1000)  # возвращает равномерно распределённые числа в данном интервале

a0 = 1
a1 = 0.5
a2 = 1.5
w0 = 0.017
w1 = 0.072
w2 = 0.002


data_1 = (np.random.random(1000))
# noise = [1*x for x in data_1]  # шум

# T= 2*pi
f1 = a0*sin(2 * pi * w0 * x + 2 * pi * 0.00) + a1*sin(2 * pi * w1 * x + 2 * pi * 0.00) + a2*cos(2 * pi * w2 * x + 2 * pi * 0.00) + [1*x for x in np.random.random(1000)]
f2 = a0*sin(2 * pi * w0 * x + 2 * pi * 0.00425) + a1*sin(2 * pi * w1 * x + 2 * pi * 0.00425) + a2*cos(2 * pi * w2 * x + 2 * pi * 0.00425) + [x for x in np.random.random(1000)]
f3 = a0*sin(2 * pi * w0 * x + 2 * pi * 0.0085) + a1*sin(2 * pi * w1 * x + 2 * pi * 0.0085) + a2*cos(2 * pi * w2 * x + 2 * pi * 0.0085) + [x for x in np.random.random(1000)]
f4 = a0*sin(2 * pi * w0 * x + 2 * pi * 0.01275) + a1*sin(2 * pi * w1 * x + 2 * pi * 0.01275) + a2*cos(2 * pi * w2 * x + 2 * pi * 0.01275) + [x for x in np.random.random(1000)]

nrows, ncols = 4, 1
figsize = [10, 7]
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
fig.suptitle('Исходный сигнал')
ax[0].plot(x, f1)
ax[0].grid(True)
ax[1].plot(x, f2)
ax[1].grid(True)
ax[2].plot(x, f3)
ax[2].grid(True)
ax[3].plot(x, f4)
ax[3].grid(True)

plt.show()

# ===================================================================================

"""   """
dataT = np.arange(len(x), 0.25)
newMass = np.zeros(len(x)*4)
for i in range(len(x)):
    newMass[4*i + 0] = f1[i]
    newMass[4*i + 1] = f2[i]
    newMass[4*i + 2] = f3[i]
    newMass[4*i + 3] = f4[i]
# ===================================================================================

spectrf = rfft(Mas - np.mean(Mas)) / (len(x))
#nu = linspace(0, 2, 2001)
nu = rfftfreq(1000, 1.)

plt.figure(figsize=(8, 5))
plt.plot(nu, np.abs(spectrf)) #возвращает частоту в Гц

axes = plt.gca()
axes.set_xlim([0.0, 0.02])
plt.xlabel('nu')
plt.ylabel('Amp')
plt.title('Спектр')
plt.grid(True)
plt.show()

