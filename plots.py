"""
	Code to for comparison and verification of codes - to make comparison plots of the relative percent error
	To read in .txt files with columns: redshift and snr
	To output: plots of the relative percent error (y-axis) versus redshift on the x-axis
	EMW February 2020
"""

import numpy as np

from tqdm import tqdm, trange
from astropy import constants as const

import math

import matplotlib.pyplot as plt


## finish imports ##

"""
	Newtonian comparisons: compare against Y&P discounting the zero-area triangles. Read in the reference file:
"""

#Y&P data:

py_data = np.loadtxt("yp_datapoints.txt")
py_z = py_data[:,0]
py_b = py_data[:,2]

#EMW data:

emw_data = np.loadtxt("newt_exclflat.txt")
emw_z = emw_data[:,0]
emw_b = emw_data[:,1]

#OU data:

#SJ data

sj_data = np.loadtxt("newt_inclflat.txt")
sj_z = sj_data[:,0]
sj_b = sj_data[:,1]

#calculate error


## plotting ##

#plt.figure(figsize=(8,8))
plt.plot(py_z, py_b, label = "P&Y",marker='o')
plt.plot(emw_z,emw_b,label= "Newt excl flat", marker='o')
plt.plot(sj_z, sj_b, label= "Newt incl flat", marker='o')
plt.ylabel("S/N")
plt.xlabel("z")
plt.legend()
plt.savefig("newt_comparisons.png")