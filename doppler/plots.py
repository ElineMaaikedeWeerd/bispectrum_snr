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

# emw_data = np.loadtxt("snr_relativistic_planck2015.txt")
# emw_z = emw_data[:,0]
# emw_bvark = emw_data[:,1]
# emw_bfixk = emw_data[:,2]
# # emw_bnewt = emw_data[:,3]


v1_data = np.loadtxt("snr_allHoverkcorr.txt")
v1_z = v1_data[:,0]
v1_snr = v1_data[:,1]
snr2 = v1_data[:,2]
snr3 = v1_data[:,3]


# v2_data = np.loadtxt("snr_fig10v2.txt")
# v2_z = v2_data[:,0]
# v2_snr = v2_data[:,1]

# v3_data = np.loadtxt("snr_fig10v3.txt")
# v3_z = v3_data[:,0]
# v3_snr = v3_data[:,1]

#SJ data

# sj_data = np.loadtxt("snr_Planck2015_Euclid_kmax_0.10hz_includeZeroAreaTriangles.dat")
# sj_z = sj_data[:,0]
# sj_bnewt = sj_data[:,1]
# sj_brel = sj_data[:,2]
# sj_cumnewt = sj_data[:,3]
# sj_cumrel = sj_data[:,4]

#calculate error

# rpe_emwsj = ((sj_b - emw_b) / sj_b) * 100
# rpe_ypemw = ((py_b - emw_b) / py_b) * 100
# rp_ypsj = ((py_b - sj_b) / py_b) * 100 
comp1 = ((np.sqrt(v1_snr) - np.sqrt(snr2) ) / np.sqrt(v1_snr)) * 100
comp2 = ((np.sqrt(v1_snr) - np.sqrt(snr3) ) / np.sqrt(v1_snr)) * 100

##
#any other stuff
##




## plotting ##

#plt.figure(figsize=(8,8))
# plt.plot(py_z, py_b, label = "P&Y",marker='o')
# plt.plot(emw_z,emw_b,label= "Newt excl flat", marker='o')
plt.plot(v1_z,np.sqrt(v1_snr),marker='o',label="Doppler")
plt.plot(v1_z,np.sqrt(snr2),marker='o',label="change Var")
plt.plot(v1_z,np.sqrt(snr3),marker='o',label="change Var+change Bg")


# plt.plot(v2_z,np.sqrt(np.cumsum(v2_snr)),marker='o',label="Q = 2, b_e = 0")
# plt.plot(v3_z,np.sqrt(np.cumsum(v3_snr)),marker='o',label="Q = 2, b_e = 1")

# plt.plot(np.arange(0.7,2.1,0.1), rpe_ypemw, label= "emw yp", marker='o')
# plt.plot(np.arange(0.7,2.1,0.1), rp_ypsj, label= "sj yp", marker='o')

plt.ylabel("S/N")
plt.xlabel("z")
plt.legend()
plt.savefig("allordersincl.png")