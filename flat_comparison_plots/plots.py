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


#with flattened and all contributions

fso_data = np.loadtxt("snr_all_inclflat.txt")
fso_z = fso_data[:,0]
fso_varmax = fso_data[:,1]
fso_fixmax = fso_data[:,2]
fso_newt = fso_data[:,3]

#wo flattened, all contri

nfso_data = np.loadtxt("snr_all_exclflat.txt")
nfso_z = nfso_data[:,0]
nfso_varmax = nfso_data[:,1]
nfso_fixmax = nfso_data[:,2]
nfso_newt = nfso_data[:,3]

#w flat, no so contr

f_data = np.loadtxt("snr_all_nosorelcontr_inclflat.txt")
f_z = f_data[:,0]
f_varmax = f_data[:,1]
f_fixmax = f_data[:,2]
f_newt = f_data[:,3]

#wo flat, no so con

nf_data = np.loadtxt("snr_all_nosorelcontr_exclflat.txt")
nf_z = nf_data[:,0]
nf_varmax = nf_data[:,1]
nf_fixmax = nf_data[:,2]
nf_newt = nf_data[:,3]


#calculate error

# rpe_emwsj = ((sj_b - emw_b) / sj_b) * 100
# rpe_ypemw = ((py_b - emw_b) / py_b) * 100
# rp_ypsj = ((py_b - sj_b) / py_b) * 100 


##
#any other stuff
##




## plotting ##

#plt.figure(figsize=(8,8))
# plt.plot(py_z, py_b, label = "P&Y",marker='o')
# plt.plot(emw_z,emw_b,label= "Newt excl flat", marker='o')

plt.plot(fso_z, np.sqrt(np.cumsum(fso_varmax)),label="N, w flat",marker='o')
plt.plot(nf_z, np.sqrt(np.cumsum(nfso_varmax)),label="N, wo flat",marker='o')


# plt.plot(np.arange(0.7,2.1,0.1), rpe_ypemw, label= "emw yp", marker='o')
# plt.plot(np.arange(0.7,2.1,0.1), rp_ypsj, label= "sj yp", marker='o')

plt.ylabel("S/N")
plt.xlabel("z")
plt.legend()
plt.savefig("cumulative.png")