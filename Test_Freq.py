#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy import (linspace, sin, cos, pi)
from numpy.fft import rfft, rfftfreq

# ===================================================================================

x=linspace(0, 1000, 1000) #возвращает равномерно распределённые числа в данном интервале

a0= 1
a1= 0.5
ω0= 0.017
ω1= 0.34

data_1 = (np.random.random(1000))
noise = [10*x for x in data_1] #шум

#T= 2*pi
f1 = a0*sin(2 * pi * ω0 * x + 2 * pi * 0.00) + [10*x for x in np.random.random(1000)]
f2 = a0*sin(2 * pi * ω0 * x + 2 * pi * 0.00425) + [10*x for x in np.random.random(1000)] 
f3 = a0*sin(2 * pi * ω0 * x + 2 * pi * 0.0085) + [10*x for x in np.random.random(1000)]
f4 = a0*sin(2 * pi * ω0 * x + 2 * pi  * 0.01275) + [10*x for x in np.random.random(1000)] 

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
spectrf = rfft(newMass - np.mean(newMass)) / (len(x))
#nu = linspace(0, 2, 2001)
nu = rfftfreq(4000, 1./4)

plt.figure(figsize=(8, 5))
plt.plot(nu, np.abs(spectrf)) #возвращает частоту в Гц

axes = plt.gca()
axes.set_xlim([0.0, 0.02])
plt.xlabel('nu')
plt.ylabel('Amp')
plt.title('Спектр')
plt.grid(True)
plt.show()

