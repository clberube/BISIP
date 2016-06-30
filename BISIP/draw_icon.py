# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:33:25 2016

@author: Charles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(1000)
fig = plt.figure()
ax = fig.add_subplot(111)
# the histogram of the data
#n, bins, patches = ax.hist(x, 50, normed=1, facecolor='orange', alpha=1)
bincenters = 0.5*(bins[1:]+bins[:-1])
# add a best fit line for the normal PDF
#y = mlab.normpdf( bincenters, mu, sigma)
#l = ax.plot(bincenters, y, 'b-', linewidth=3)
plt.plot(x, color='orange', linewidth=3)

plt.plot(x, color='blue', linewidth=1)

plt.title("BISIP", fontsize=40)
ax.grid(True)
plt.grid(None)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.setp( ax.get_yticklabels(), visible=False)
#plt.axis('off')
plt.xlabel("Bayesian inference of SIP\n parameters using MCMC simulation", fontsize=17)
plt.savefig("Icon.png", dpi=150)
plt.show()
