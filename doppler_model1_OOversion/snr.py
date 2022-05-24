import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d

import camb

from tqdm import tqdm, trange
# from astropy import constants as const

import math

import time 

from numba import jit

import profile


# Planck 2018 fiducial values for cosmological parameters 
c = 2.99792458e5
hubble = 0.6766
omegab = 0.02242 * hubble**-2 
omegac = 0.11933 * hubble**-2 
om0 = 0.3111  #omegac+omegab
H00 = 100 * hubble
Ass = 2.139e-9
nss = 0.9665
gamma = 0.545

#Set up the fiducial cosmology (CAMB)
pars = camb.CAMBparams()
pars.set_cosmology(H0=H00, ombh2=omegab*pow(hubble,2), omch2=omegac*pow(hubble,2),omk=0,mnu=0)
pars.set_dark_energy() #LCDM (default)
pars.InitPower.set_params(ns=nss, r=0, As=Ass)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

background = camb.get_background(pars)

#table of b_e and Q from doppler paper
euclid_data = np.loadtxt('snr_surveyparams.txt')

#the table from the draft  has columns z, b_e, Q, n_g, V, sigma
z_euclid = euclid_data[:,0]
be_euclid = interp1d(z_euclid, euclid_data[:,1])
Q_euclid = interp1d(z_euclid,  euclid_data[:,2])
ngt_euclid = interp1d(z_euclid,1e-3 * euclid_data[:,3])
vt_euclid = interp1d(z_euclid, 1e9 * euclid_data[:,4])
sigma_euclid = interp1d(z_euclid, euclid_data[:,5])


class powerspectrum:
	def __init__(self,Z=1,k=None):
		self.Z = Z 
		self.k = k 

	def set_k(self,k):
		self.k = k 

	def P_matter(self):
		#linear matter power spectrum
		pars.set_matter_power(redshifts=[self.Z], kmax=2)
		pars.NonLinear = camb.model.NonLinear_none
		results = camb.get_results(pars)
		kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints = 10000)
		s8 = np.array(results.get_sigma8())
		#Pm in, at z=Z
		Pmz = interp1d(kh, (pk[0]))
		return Pmz(self.k)

	def KN1(self,mu):
		B1 =  0.9 + 0.4 * self.Z
		omz = om0 * np.power(1 + self.Z, 3) / (om0 * np.power(1 + self.Z,3) + 1 - om0)
		f = np.power(omz,gamma)
		return B1 + f * mu**2

	def P_twiddle(self,mu,sigma,ng_euclid,damp_on_Ptw):
		"""
			P-twiddle: galaxy power spectrum with noise term. Damping on this can be turned on and off
			setting damp_on_Ptw to True or False
		"""
		if damp_on_Ptw:
			damp_term = np.exp(- (1/2) * ((self.k * mu * sigma)**2))
		else:
			damp_term = 1
		return self.KN1(mu)**2  * self.P_matter() * damp_term + (1/ng_euclid)


class bispectrum:
	def __init__(self,Z=1,k={1:0.01,2:0.01,3:0.01,"costheta": -0.5},Newtonian=False,mu=-1,phi=0):
		self.Z = Z 
		self.k = k 
		self.set_mu(mu,phi)
		self.Newtonian = Newtonian
		self.camb_res()
		self.get_growth()
		self.calc_cosmology_params()
		self.set_beta_coeffs()

	def set_k(self,k):
		self.k = k 

	# def set_mu(self,mu):
	# 	self.mu = mu 

	def set_mu(self,MU_1,PHI):
		"""
		Function to return mu1,mu2,mu3 when given mu1 and phi, and k as dict. Returns dict
		"""
		self.mu = {1:MU_1}
		self.mu[2] = self.mu[1]*(self.k["costheta"]) + np.sqrt(1.0-self.mu[1]**2) * np.sqrt(abs(1 - (self.k["costheta"])**2)) *np.cos(PHI)
		self.mu[3] = - (self.k[1] / self.k[3]) * self.mu[1] - (self.k[2] / self.k[3]) * self.mu[2]

	def camb_res(self):
		pars.set_matter_power(redshifts=[self.Z], kmax=2)
		pars.NonLinear = camb.model.NonLinear_none
		self.results = camb.get_results(pars)
		# print(self.results)

	#Define E(z) = H(z)/H0
	def Ez(self):
		self.EZ = np.sqrt(1 - om0 + om0 * np.power(1 + self.Z,3))

	#Define the comoving distance
	def drdz(self):
		return (c / H00) / self.EZ

	def rcom(self):
		self.rcom = sp.integrate.romberg(self.drdz(),0,self.Z)

	#Define the growth function in LCDM (growth rate)
	def get_growth(self):
		omz = om0 * np.power(1 + self.Z, 3) / (om0 * np.power(1 + self.Z,3) + 1 - om0)
		self.growthrate = np.power(omz,gamma)

	#Get the growth factor
	def Dg_dz(self):
		return get_growth(self.Z) / (1 + self.Z)

	def Dgz(self):
		ans = integrate.romberg(Dg_dz(), 0, self.Z)
		self.growthfactor = np.exp(-ans)

	#power spectrum at redshift Z for dictionary element k[i]
	def P_matter(self,i):
		#linear matter power spectrum
		kk = self.k[i]
		Pmz = powerspectrum(self.Z,kk)
		return Pmz.P_matter()

	# def P_galaxy(self,i):
	# 	#galaxy power spectrum 
	# 	return self.KN1(i)**2 * self.P_matter(i)

	def get_cosinus(self,i,j,k):
		"""
			Returns cosine of angle between entry i and entry j of dict k
		"""
		perms = [1,2,3]
		assert i!=j
		assert i in perms
		assert j in perms
		perms.remove(i)
		perms.remove(j)
		l = perms[0]

		return 0.5 * (self.k[l]**2 - self.k[i]**2 - self.k[j]**2) / (self.k[i] * self.k[j])

	def calc_cosmology_params(self):
		Z = self.Z
		#Calculate all params at redshift Z
		Hu = self.results.h_of_z(Z) * (1 / (1 + Z)) / hubble 
		H0 = self.results.h_of_z(0) / hubble
		om_m0 = om0
		om_m = om_m0 * (H0**2 / Hu**2) * (1+Z)
		f = self.growthrate
		df = Hu * ( (1/2) * (3 * om_m - 4) * f - f**2 + (3/2) * om_m)
		dHu = H0**2 * (-(1/2)* (1+Z) * om_m0 + (1 / (1+Z) )**2 * (1 - om_m0)) 
		ddHu = H0**2 * ( (1/2) * Hu * (1+Z) * om_m0 + (1 / (1+Z))**2 * 2 * Hu * (1 - om_m0) ) 
		chi = self.results.angular_diameter_distance(Z) * (1 + Z) * hubble 
		cap_L = 1
		partdQ=0
		B1 =  0.9 + 0.4 * Z
		db1 = -0.4 * Hu * (1 + Z)
		B2 = -0.741 - 0.125 * Z + 0.123 * Z**2 + 0.00637 * Z**3
		b_e = be_euclid(Z)
		db_e=0
		bs = 0.0409 - 0.199 * Z - 0.0166 * Z**2 + 0.00268 * Z**3
		Q =  Q_euclid(Z)
		dQ= 0
		gamma1 = Hu* (f * (b_e - 2*Q -( 2 * (1 - Q) / (chi * Hu)) - (dHu / Hu**2)))
		gamma2 = Hu**2 * (f * (3 - b_e) + (3/2) * om_m * (2 + b_e - f- 4 * Q - (2 * (1 - Q) / (chi * Hu)) - (dHu / Hu**2) ))
		partdb1 = 0
		self.cosmology_params = {"Hu":Hu,"H0":H0,"om_m0":om_m0,"om_m":om_m,"f":f,"df":df,"dHu":dHu,"ddHu":ddHu,"chi":chi,"cap_L":cap_L,"partdQ":partdQ,"B1":B1,"db1":db1,"B2":B2,"b_e":b_e,"db_e":db_e,"bs":bs,"Q":Q,"dQ":dQ,"gamma1":gamma1,"gamma2":gamma2,"partdb1":partdb1}

	def set_beta_coeffs(self):
		"""
			Function that relies on global variables to return beta coefficients
			from papers
		"""
		beta = np.zeros(20)
		beta[1] = self.cosmology_params["Hu"]**4 * ( (9 / 4) * self.cosmology_params["om_m"]**2 * (6 - 2 * self.cosmology_params["f"]* (2 * self.cosmology_params["b_e"] - 4 * self.cosmology_params["Q"] - (4 * (1 - self.cosmology_params["Q"]))/(self.cosmology_params["chi"] * self.cosmology_params["Hu"]) - (2 * self.cosmology_params["dHu"])/(self.cosmology_params["Hu"]**2)) - (2 * self.cosmology_params["df"])/(self.cosmology_params["Hu"]) + self.cosmology_params["b_e"]**2 + 5 * self.cosmology_params["b_e"] - 8 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] + 4 * self.cosmology_params["Q"] + 16 * self.cosmology_params["Q"]**2 - 16 * self.cosmology_params["partdQ"] - 8 * self.cosmology_params["dQ"] / self.cosmology_params["Hu"] + self.cosmology_params["db_e"] / self.cosmology_params["Hu"] + (2 / (self.cosmology_params["chi"]**2 * self.cosmology_params["Hu"]**2)) * (1 - self.cosmology_params["Q"] + 2 * self.cosmology_params["Q"]**2 - 2 * self.cosmology_params["partdQ"]) - (2 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) * (3 + 2 * self.cosmology_params["b_e"] - 2 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] - 3 * self.cosmology_params["Q"] + 8 * self.cosmology_params["Q"]**2 - (3 * self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) * (1 - self.cosmology_params["Q"]) - 8 * self.cosmology_params["partdQ"] - 2 * self.cosmology_params["dQ"] / self.cosmology_params["Hu"]) + (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) * (-7 - 2 * self.cosmology_params["b_e"] + 8 * self.cosmology_params["Q"] + (3 * self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)) - (self.cosmology_params["ddHu"] / self.cosmology_params["Hu"]**3)) + ((3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"]) * (5 - 2 * self.cosmology_params["f"]* (4 - self.cosmology_params["b_e"]) + (2 * self.cosmology_params["df"]/ self.cosmology_params["Hu"]) + 2 * self.cosmology_params["b_e"] * (5 + ((2 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"]))) - (2 * self.cosmology_params["db_e"] / self.cosmology_params["Hu"]) - 2 * self.cosmology_params["b_e"]**2 + 8 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] - 28 * self.cosmology_params["Q"] - (14 * (1 - self.cosmology_params["Q"]) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - 3 * self.cosmology_params["dHu"] / self.cosmology_params["Hu"]**2 + 4 * (2 - (1 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"]))) * (self.cosmology_params["dQ"] / self.cosmology_params["Hu"]) ) + ((3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"]**2) * (-2 + 2 * self.cosmology_params["f"]- self.cosmology_params["b_e"] + 4 * self.cosmology_params["Q"] + (2 * (1 - self.cosmology_params["Q"]) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) + (3 * self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) ) + self.cosmology_params["f"]**2 * (12 - 7 * self.cosmology_params["b_e"] + self.cosmology_params["b_e"]**2 + (self.cosmology_params["db_e"] / self.cosmology_params["Hu"]) + (self.cosmology_params["b_e"] - 3) * (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)) - (3 / 2) * self.cosmology_params["om_m"] * (self.cosmology_params["df"]/ self.cosmology_params["Hu"]))
		beta[2] = self.cosmology_params["Hu"]**4 * ((9 / 2) * self.cosmology_params["om_m"]**2 * (-1 + self.cosmology_params["b_e"] - 2 * self.cosmology_params["Q"] - ((2 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) ) + 3 * self.cosmology_params["om_m"] * self.cosmology_params["f"] * (-1 + 2 * self.cosmology_params["f"]- self.cosmology_params["b_e"] + 4 * self.cosmology_params["Q"] + (2 * (1 - self.cosmology_params["Q"]) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) + (3 * self.cosmology_params["dHu"] / self.cosmology_params["Hu"]**2) )  + 3 * self.cosmology_params["om_m"] * self.cosmology_params["f"]**2 * (-1 + self.cosmology_params["b_e"] - 2 * self.cosmology_params["Q"] - ((2 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2))  + 3 * self.cosmology_params["om_m"] * (self.cosmology_params["df"]/ self.cosmology_params["Hu"]))
		beta[3] = self.cosmology_params["Hu"]**3 * ((9 / 4) * self.cosmology_params["om_m"]**2 * (self.cosmology_params["f"]- 2 + 2 * self.cosmology_params["Q"]) + (3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"]* (-2 - self.cosmology_params["f"]* (-3 + self.cosmology_params["f"]+ 2 * self.cosmology_params["b_e"] - 3 * self.cosmology_params["Q"] - ((4 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (2 * self.cosmology_params["dHu"] / self.cosmology_params["Hu"]**2)) - (self.cosmology_params["df"]/ self.cosmology_params["Hu"]) + 3 * self.cosmology_params["b_e"] + self.cosmology_params["b_e"]**2 - 6 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] + 4 * self.cosmology_params["Q"] + 8 * self.cosmology_params["Q"]**2 - 8 *self.cosmology_params["partdQ"] - 6 * (self.cosmology_params["dQ"] / self.cosmology_params["Hu"]) + (self.cosmology_params["db_e"] / self.cosmology_params["Hu"]) + (2 / (self.cosmology_params["chi"]**2 * self.cosmology_params["Hu"]**2)) * (1 - self.cosmology_params["Q"] + 2 * self.cosmology_params["Q"]**2 - 2 * self.cosmology_params["partdQ"]) + (2 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) * (-1 - 2 * self.cosmology_params["b_e"] + 2 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] + self.cosmology_params["Q"] - 6 * self.cosmology_params["Q"]**2 + ( 3 * self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) * (1 - self.cosmology_params["Q"]) + 6 * self.cosmology_params["partdQ"] + 2 * (self.cosmology_params["dQ"] / self.cosmology_params["Hu"]) ) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) * (3 + 2 * self.cosmology_params["b_e"] - 6 * self.cosmology_params["Q"] - (3 * self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)) - (self.cosmology_params["ddHu"] / self.cosmology_params["Hu"]**3)) + self.cosmology_params["f"]**2 * (-3 + 2 * self.cosmology_params["b_e"] * (2 + ((1 - self.cosmology_params["Q"]) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"]))) - self.cosmology_params["b_e"]**2 + 2 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] - 6 * self.cosmology_params["Q"] - (self.cosmology_params["db_e"] / self.cosmology_params["Hu"]) - ((6 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) + 2 * (1 - (1 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"]))) * (self.cosmology_params["dQ"] / self.cosmology_params["Hu"]) ))
		beta[4] = (self.cosmology_params["Hu"]**3 * ((9 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"]* (- self.cosmology_params["b_e"] + 2 * self.cosmology_params["Q"] + (2 * (1 - self.cosmology_params["Q"]) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) + (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2))))
		beta[5] = (self.cosmology_params["Hu"]**3 * ( 3 * self.cosmology_params["om_m"] * self.cosmology_params["f"]* (self.cosmology_params["b_e"] - 2 * self.cosmology_params["Q"] - (2 * (1 - self.cosmology_params["Q"]) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)) ))
		beta[6] = self.cosmology_params["Hu"]**2 * ((3 / 2) * self.cosmology_params["om_m"] * (2 - 2 * self.cosmology_params["f"]+ self.cosmology_params["b_e"] - 4 * self.cosmology_params["Q"] - ((2 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)))
		beta[7] = self.cosmology_params["Hu"]**2 * (self.cosmology_params["f"]* (3 - self.cosmology_params["b_e"]))
		beta[8] = self.cosmology_params["Hu"]**2 * (3 * self.cosmology_params["om_m"] * self.cosmology_params["f"]* (2 - self.cosmology_params["f"]- 2 * self.cosmology_params["Q"]) + self.cosmology_params["f"]**2 * (4 + self.cosmology_params["b_e"] - self.cosmology_params["b_e"]**2 + 4 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] - 6 * self.cosmology_params["Q"] - 4 * self.cosmology_params["Q"]**2 + 4 * self.cosmology_params["partdQ"] + 4 * (self.cosmology_params["dQ"] / self.cosmology_params["Hu"]) -(self.cosmology_params["db_e"] / self.cosmology_params["Hu"]) - (2 / (self.cosmology_params["chi"]**2 * self.cosmology_params["Hu"]**2)) * (1 - self.cosmology_params["Q"] + 2 * self.cosmology_params["Q"]**2 - 2 * self.cosmology_params["partdQ"]) - (2 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) * (3 - 2 * self.cosmology_params["b_e"] + 2 * self.cosmology_params["b_e"] * self.cosmology_params["Q"] - self.cosmology_params["Q"] - 4 * self.cosmology_params["Q"]**2 + ((3 * self.cosmology_params["dHu"]) / (self.cosmology_params["Hu"]**2)) * (1 - self.cosmology_params["Q"]) + 4 * self.cosmology_params["partdQ"] + 2 * (self.cosmology_params["dQ"] / self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) * (3 - 2 * self.cosmology_params["b_e"] + 4 * self.cosmology_params["Q"] + ((3 * self.cosmology_params["dHu"]) / (self.cosmology_params["Hu"]**2))) + (self.cosmology_params["ddHu"] / self.cosmology_params["Hu"]**3)))
		beta[9] = (self.cosmology_params["Hu"]**2 * ( -(9 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"]))
		beta[10] = self.cosmology_params["Hu"]**2 * (3 * self.cosmology_params["om_m"] * self.cosmology_params["f"])
		beta[11] = self.cosmology_params["Hu"]**2 * ( (3/2 ) * self.cosmology_params["om_m"] * (1 + 2 * self.cosmology_params["f"]/ (3 * self.cosmology_params["om_m"])) + 3 * self.cosmology_params["om_m"] * self.cosmology_params["f"]- self.cosmology_params["f"]**2 * (-1 + self.cosmology_params["b_e"] - 2 * self.cosmology_params["Q"] - ((2 * (1 + self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)))
		beta[12] = self.cosmology_params["Hu"]**2 * ( -3 * self.cosmology_params["om_m"] * (1 + 2 * self.cosmology_params["f"]/ (3 * self.cosmology_params["om_m"])) - self.cosmology_params["f"]* ( self.cosmology_params["B1"] * (self.cosmology_params["f"]- 3 + self.cosmology_params["b_e"]) + (self.cosmology_params["db1"] / self.cosmology_params["Hu"]) ) + (3 / 2) * self.cosmology_params["om_m"] * (self.cosmology_params["B1"] * (2 + self.cosmology_params["b_e"] - 4 * self.cosmology_params["Q"] - 2 * ((1 - self.cosmology_params["Q"])/(self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2) ) + self.cosmology_params["db1"] / self.cosmology_params["Hu"] + 2 * (2 - (1 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) ) * self.cosmology_params["partdb1"] ) )    
		beta[13] = self.cosmology_params["Hu"]**2 * (( (9 / 4) * self.cosmology_params["om_m"]**2 + (3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"]* (1 - (2 * self.cosmology_params["f"]) + 2 * self.cosmology_params["b_e"] - 6 * self.cosmology_params["Q"] - ((4 * (1 - self.cosmology_params["Q"]))/(self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - ((3 * self.cosmology_params["dHu"]) / self.cosmology_params["Hu"]**2) ) ) + ( self.cosmology_params["f"]**2 * (3 - self.cosmology_params["b_e"]) ) )
		beta[14] = self.cosmology_params["Hu"] * ( - (3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["B1"] )
		beta[15] = self.cosmology_params["Hu"] * 2 * self.cosmology_params["f"]**2
		beta[16] = self.cosmology_params["Hu"] * (self.cosmology_params["f"]* (self.cosmology_params["B1"] * (self.cosmology_params["f"]+ self.cosmology_params["b_e"] - 2 * self.cosmology_params["Q"] - ((2 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)) + (self.cosmology_params["db1"] / self.cosmology_params["Hu"]) + 2 * (1 - (1 / (self.cosmology_params["chi"] * self.cosmology_params["Hu"]))) * self.cosmology_params["partdb1"] ))
		beta[17] = self.cosmology_params["Hu"] * (- (3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"])
		beta[18] = self.cosmology_params["Hu"] * ( (3 / 2) * self.cosmology_params["om_m"] * self.cosmology_params["f"] - self.cosmology_params["f"]**2 * (3 - 2 * self.cosmology_params["b_e"] + 4 * self.cosmology_params["Q"] + ((4 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) + (3 * self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)) )
		beta[19] = self.cosmology_params["Hu"] * (self.cosmology_params["f"]* (self.cosmology_params["b_e"] - 2 * self.cosmology_params["Q"] - ((2 * (1 - self.cosmology_params["Q"])) / (self.cosmology_params["chi"] * self.cosmology_params["Hu"])) - (self.cosmology_params["dHu"]/ self.cosmology_params["Hu"]**2)))

		self.beta = beta


	#doppler snr: only O(1/k) corrections included

	def E(self,i,j,l):
		"""
			Kernel E_2 from paper
		"""
		cosinus = self.get_cosinus(i,j,self.k)
		return ((self.k[i]**2 * self.k[j]**2) / (self.k[l]**4) ) * (3 + 2 * cosinus * (self.k[i]/self.k[j] + self.k[j]/self.k[i]) + cosinus**2  )

	def F(self,i,j,l):
		"""
			Kernel F_2 from paper
		"""
		cosinus = self.get_cosinus(i,j,self.k)
		return 10/7 + cosinus * (self.k[i]/self.k[j] + self.k[j]/self.k[i]) + (1 - 3/7) * cosinus**2

	def G(self,i,j,l):
		"""
			Kernel G_2 from paper
		"""
		cosinus = self.get_cosinus(i,j,self.k)
		return 6/7 + cosinus * (self.k[i]/self.k[j] + self.k[j]/self.k[i]) + (2 - 6/7) * cosinus**2


	#First order Newtonian, GR Fourier space kernels,
	def KN1(self,i):
		return self.cosmology_params["B1"] + self.cosmology_params["f"] * self.mu[i]**2

	def KGR1(self,i):
		return (self.cosmology_params["gamma1"]/self.k[i]) * 1j * self.mu[i] 

	#Second order Newtonian, GR Fourier space kernels
	def KN2(self,i,j,l):
		cosinus = self.get_cosinus(i,j,self.k)
		F_f = self.F(i,j,l)
		G_f = self.G(i,j,l)
		p1 = self.cosmology_params["B1"] * F_f + self.cosmology_params["B2"] + self.cosmology_params["f"] * G_f * self.mu[l]**2
		p2 = self.cosmology_params["bs"] * (cosinus**2 - 1/3)
		p3 = self.cosmology_params["f"]**2 * (self.mu[i] * self.mu[j])/(self.k[i] * self.k[j]) * (self.mu[i] * self.k[i] + self.mu[j] * self.k[j])**2
		p4 = self.cosmology_params["B1"] * self.cosmology_params["f"]/(self.k[i] * self.k[j]) * ((self.mu[i]**2 + self.mu[j]**2) * self.k[i] * self.k[j] + self.mu[i] * self.mu[j] * (self.k[i]**2 + self.k[j]**2))
		return p1 + p2 + p3 + p4

	def KGR2(self,i,j,l):
		cosinus = self.get_cosinus(i,j,self.k)
		E_f = self.E(i,j,l)
		F_f = self.F(i,j,l)
		G_f = self.G(i,j,l)
		k_prod = self.k[i]**2 * self.k[j]**2
		# p1 = beta[1] + E_f * beta[2]
		# p2 = 1j* ((mu[i] * k[i] + mu[j] * k[j]) * beta[3] + mu[l] * k[l] * (beta[4] + E_f * beta[5]) )
		# p3 = k_prod/(k[l]**2) * (F_f * beta[6] + G_f * beta[7]) + (mu[i] * k[i] * mu[j] * k[j])*beta[8]
		# p4 = mu[l]**2 * k[l]**2 * (beta[9] + E_f * beta[10]) + (k[i] * k[j] * cosinus) * beta[11]
		# p5 = (k[i]**2 + k[j]**2) * beta[12] + (mu[i]**2 * k[i]**2 + mu[j]**2 * k[j]**2) * beta[13]
		p_comp1 = (self.mu[i] * self.k[i]**3 + self.mu[j] * self.k[j]**3) * self.beta[14]
		p_comp2 = (self.mu[i] * self.k[i] + self.mu[j] * self.k[j]) * self.k[i] * self.k[j] * cosinus * self.beta[15]
		p_comp3 = self.k[i] * self.k[j] * (self.mu[i] * self.k[j] + self.mu[j] * self.k[i]) * self.beta[16]
		p_comp4 = (self.mu[i]**3 * self.k[i]**3 + self.mu[j]**3 * self.k[j]**3) * self.beta[17]
		p_comp5 = self.mu[i] * self.mu[j] * self.k[i] * self.k[j] * (self.mu[i] * self.k[i] + self.mu[j] * self.k[j]) * self.beta[18]
		p_comp6 = self.mu[l] * k_prod/self.k[l] * G_f * self.beta[19]
		# real = p1 + p2 + p3 + p4 + p5
		comp = p_comp1 + p_comp2 + p_comp3 + p_comp4 + p_comp5 + p_comp6
		return  (1/k_prod) * (1j * comp)
	
	def B_perm(self,i,j,l):
		"""
			General form of one of the cyclic permutations. Newtonian is boolean,
			if set to True this will return purely Newtonian, if False this will 
			give Doppler/GR (without newtonian!!)
		"""
		if self.Newtonian:
			return self.KN1(i) * self.KN1(j) * self.KN2(i,j,l) *  self.P_matter(i) * self.P_matter(j)
		else:
			T1 = self.KGR1(i) * self.KN1(j) *  self.KN2(i,j,l)
			T2 = self.KGR1(j) * self.KN1(i) *  self.KN2(i,j,l)
			T3 = self.KN1(i) * self.KN1(j) * self.KGR2(i,j,l)
			return (T1 + T2 + T3) * self.P_matter(i) * self.P_matter(j)

	#full bispectrum w all orders H/k
	# def B_perm(i,j,l,k,mu,B1,B2,gamma1,gamma2,beta,f):
	# 	"""
	# 		General form of one of the cyclic permutations. Newtonian is boolean,
	# 		if set to True this will return purely Newtonian, if False this will 
	# 		give Doppler H/k 
	# 	"""
	# 	if Newtonian:
	# 		return KN1(i,mu,B1,f) * KN1(j,mu,B1,f) * KN2(i,j,l,k,mu,B1,B2,f) * Pm(i,k) * Pm(j,k)
	# 	else:
	# 		p1 = KN1(i,mu,B1,f) + KGR1(i,mu,k,gamma1,gamma2)
	# 		p2 = KN1(j,mu,B1,f) + KGR1(j,mu,k,gamma1,gamma2)
	# 		p3 = KN2(i,j,l,k,mu,B1,B2,f) + KGR2(i,j,l,k,mu,beta)
	# 		return (p1 * p2 * p3) * Pm(i,k) * Pm(j,k)

	def B_full(self):
		"""
			Full bispectrum
		"""
		return (self.B_perm(1,2,3) + self.B_perm(1,3,2) + self.B_perm(2,3,1)) 



	# def comp_bisp(self):
	# 	self.Ez()
	# 	self.rcom()
	# 	self.get_growth()
	# 	self.Dgz()
	# 	self.set_beta_coeffs()

class SNR:
	def __init__(self,Z=1,damp=True,Newtonian=False,damp_on_Ptw=True,kmax_zdep=True):
		self.Z = Z 
		self.damp = damp 
		self.Newtonian = Newtonian
		self.damp_on_Ptw = damp_on_Ptw
		self.kmax_zdep = kmax_zdep
		self.sigma = sigma_euclid(Z)
		self.ng_euclid = ngt_euclid(Z)
		self.v_euclid = vt_euclid(Z)
		self.k_fundamental = (2 * np.pi) / np.power(self.v_euclid, 1/3)
		#step sizes
		self.deltamu = 0.04
		self.deltaphi = np.pi/25
		self.mu_range = 2.0
		self.phi_range = 2 * np.pi
		self.deltaz = 0.1
		self.powerspectrum = powerspectrum(Z=Z,k=0.01)
		self.bispectrum = bispectrum(Z=Z,mu=-1,phi=0)

	def get_costheta(self,k1,k2,k3):
	    """
	    Function to get angle between two wavevectors
	    """
	    x =  0.5 * ( k3**2 - (k1**2 + k2**2))/(k1 * k2)
	    return x 

	def get_mus(self,MU_1,PHI,k):
		"""
		Function to return mu1,mu2,mu3 when given mu1 and phi, and k as dict. Returns dict
		"""
		mu = {1:MU_1}
		mu[2]=mu[1]*(k["costheta"]) + np.sqrt(1.0-mu[1]**2) * np.sqrt(abs(1 - (k["costheta"])**2)) *np.cos(PHI)
		mu[3] = - (k[1] / k[3]) * mu[1] - (k[2] / k[3]) * mu[2]
		return mu


	def P_twiddle(self,i,k,mu):
	# 	"""
	# 		P-twiddle: galaxy power spectrum with noise term. Damping on this can be turned on and off
	# 		setting damp_on_Ptw to True or False
	# 	"""
		self.powerspectrum.set_k(k[i])
		return self.powerspectrum.P_twiddle(mu[i],self.sigma,self.ng_euclid,self.damp_on_Ptw)

	# def P_galaxy(i,k,mu,B1,f):
	# 	"""
	# 		P_galaxy: galaxy power spectrum with damping
	# 	"""
	# 	return KN1(i,mu,B1,f)**2 * Pm(i,k) * np.exp(- (1/2) * ((k[i] * mu[i] * self.sigma)**2))


	def s_B(self,k):
		"""
			s_B takes a dictionary and returns an integer, to take symmetry/overcounting into account
			in the Var[B]
		"""
		if (math.isclose(k[1],k[2],abs_tol=1e-8) and math.isclose(k[2],k[3],abs_tol=1e-8)):
			return 6
		elif (math.isclose(k[1],k[2],abs_tol=1e-8) or math.isclose(k[1],k[3],abs_tol=1e-8) or math.isclose(k[2],k[3],abs_tol=1e-8)):
			return 2
		else:
			return 1


	def arr_func(self,k,mu1,phis):
		"""
			this is a function that works with lists/arrays (of k, mu, phi) and quickly returns
			the SNR^2. 
			mu : calculates set of mu_i 
			bisp : calculates full bispectrum
			varb_num : numerator of Var[B] expression in 1911.02398
			varb_den : denominator of Var[b] in 1911.02398
			Square bispectrum, divide by Var[B] = multiply by denominator, divide by numerator. Sum result
		"""

		self.bispectrum.set_k(k)
		self.bispectrum.set_mu(mu1,phis) 
		bisp = self.bispectrum.B_full()
		varb_num =  s_B(k) * np.pi * np.power(self.k_fundamental,3) * self.mu_range * self.phi_range * self.P_twiddle(1,k,self.bispectrum.mu) * self.P_twiddle(2,k,self.bispectrum.mu) * self.P_twiddle(3,k,self.bispectrum.mu)		
		varb_den = k[1] * k[2] * k[3] * np.power(self.k_fundamental,3) * self.deltamu * self.deltaphi
		res = (abs(bisp)**2) * varb_den / varb_num
		return res.sum()


	def set_kmax(self):
		if self.kmax_zdep:
			return 0.1 * (1 + self.Z)**((2/(2 + nss)))
		else:
			return 0.15

	def calculate_snr(self):
		mu_bins = np.arange(-1,1,self.deltamu)
		phi_bins = np.arange(0,2*np.pi,self.deltaphi)
		mu1 = np.tile(mu_bins,(50,1))
		phis = np.tile(phi_bins,(50,1)).T
		kmin = self.k_fundamental
		deltak = self.k_fundamental
		kmax = self.set_kmax()
		#binning k between kmin and kmax with deltak steps
		k_bins = np.arange(kmin,kmax+deltak,deltak)
		snr=0
		klist=[]
		#going through the bins, checking triangle conditions, appending valid triangles as dict
		for k1 in k_bins:
			for k2 in k_bins[k_bins<=k1]:
				for k3 in k_bins[k_bins<=k2]:  
					if (k1 - k2 - k3 <= 1e-8):
						k = {1:k1, 2:k2, 3:k3, "costheta":self.get_costheta(k1,k2,k3)} 
						klist.append(k)
		#calculating snr^2 
		for k in tqdm(klist):
			snr += self.arr_func(k,mu1,phis)
		return snr

		

	
if __name__ == "__main__":


	bub = SNR(Z=0.7)
	print(bub.calculate_snr())

