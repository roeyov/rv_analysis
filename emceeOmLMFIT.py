import emcee
# import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import pandas as pd
# from astropy.visualization.units import quantity_support
import os
import sys
import thejoker as tj
import math
from astropy.io import ascii
import lmfit
from scipy.interpolate import interp1d
import matplotlib.pylab as pylab

import corner
import arviz as az
import astropy.coordinates as coord
import make_RVs

# import importlib
# importlib.reload(make_RVs)
T0 = 0
P = 250
N_OF_PS = 15
ECC = 0.3
OMEGA_deg = 30
OMEGA_rad = OMEGA_deg * np.pi / 180
K1 = 7
K2 = 7
GAMMA = 10
NRV = 25
sig_rv = 3.0

# np.random.seed(2993)
# rnd = np.random.default_rng(seed=5492)
# def                         out_single_and_plot(t0, p,  ecc, omega, k1, k2, gamma, nrv, sig_rv, plot=False):
rvs_1, ts, errs_v1 = make_RVs.out_single_and_plot(T0, P, ECC, OMEGA_rad, K1, K2, GAMMA, NRV, sig_rv, N_OF_PS, True)
astropy_rvs = u.quantity.Quantity(rvs_1, unit=u.km / u.s)
astropy_err_rvs = u.quantity.Quantity(np.ones(NRV) * sig_rv, unit=u.km / u.s)
t = Time(ts + 60347, format="mjd", scale="tcb")
# df = pd.DataFrame(np.stack([ts, rvs_1, errs_v1]).T, columns= [ 'ts', "rvs", 'errs_v1'])
# df.to_csv("/media/sf_Roey\'s/Masters/General/Scripts/scriptsOut/tomer_df.csv")
data = tj.RVData(t=t, rv=astropy_rvs, rv_err=astropy_err_rvs)
_ = data.plot()
plt.show()

params = {'legend.fontsize': 'large',
          'figure.figsize': (12, 4),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
locleg = 'upper left'

CompNum = 1

BinTime = .1
BinTime = 20.

sys.setrecursionlimit(int(1E6))

## Program for fitting polarimetric data, based on Robert+ 1992, ApJ, 397, 277
## Input: orbital parameters (with period/period interval), polarimetric data
## Output: best-fitting parameters:
# BigOmega = orientation of line of nodes
# inc = inclination angle
# taustar = total optical depth of free electrons
# Q0 = interstellar component of Q
# U0 = "" of U
# gamma = exponent describing the parameter tau3, typically 1. (see paper above)
# file: schnurrdat-Us-phase.txt and schnurrdat-Qs-phase.txt
# file: best-fitting-QU.txt - with phases, Q, and U
# file: chi2period.txt: Showing minimum chi2 as function of period


maxitr = 1E2
tolerance = 1E-10
ftolerance = 1E-10

# *********************** USER INPUT ***********************
velos = {'ts': ts,
         'rvs': rvs_1,
         'errs_v1': np.ones(ts.shape) * sig_rv}
hjds1 = np.array(velos['ts'])
v1s = np.array(velos['rvs'])
errv1s = np.abs(np.array(velos['errs_v1']))
# MiniMethod = 'leastsqr'
# MiniMethod = 'least_squares'
# MiniMethod = 'nelder'
MiniMethod = 'differential_evolution'
# MiniMethod = 'lbfgsb'
# MiniMethod = 'powell'
T0Ini = 0.
Pini = 241
# Please note initial values for parameters, bounds, and whether they should vary or not
params = lmfit.Parameters()
# params.add('P',   value=1.596,  min=1.590, max=1.700, vary=True)
params.add('P', value=Pini, min=221, max=261., vary=True)
params.add('Gamma1', value=0, min=-300., max=300., vary=True)
# params.add('Gamma2',   value=0,  min=-300., max=300.,  vary=True)
params.add('K1', value=200, min=0., max=400., vary=True)
# if CompNum==2:
# params.add('K2', value=10.,  min=0., max=50., vary=True)
params.add('Omega', value=90., min=-180., max=180., vary=True)
params.add('ecc', value=0., min=.0, max=.9, vary=True)
# params.add('ShiftSec', value= 0.,  min=-400, max=400, vary=False)
params.add('T0', value=T0Ini, min=T0Ini - Pini * 0.5, max=T0Ini + Pini * 0.5, vary=True)
# *********************** END OF USER INPUT ***********************
# Initialization
chi2list = []


def NuDerIntr(phi, ecc):
    phi[phi > 0.5] -= 1
    phi[phi < -0.5] += 1
    phi[phi < -0.49] = -0.49
    phi[phi > 0.49] = 0.49
    phscontBin = 0.001
    phscont = np.arange(-1., 1., phscontBin)
    Ms = 2 * np.pi * phscont
    Es = Kepler(np.pi, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    nuscont = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    nuscontder = np.concatenate((np.array([1]), np.diff(nuscont))) / phscontBin
    # DerFunc = interp1d(phscont, nuscontder, bounds_error=False, fill_value=1., kind=intr_kind)(phi)
    return interp1d(phscont, nuscontder)(phi)


# For converting Mean anomalies to eccentric anomalies (M-->E)
def Kepler(E, M, ecc, itrnum=1):
    # print itrnum
    E2 = (M - ecc * (E * np.cos(E) - np.sin(E))) / (1. - ecc * np.cos(E))
    eps = np.abs(E2 - E)
    if np.all(eps < 1E-5):
        return E2
    else:
        return Kepler(E2, M, ecc, itrnum=1)
        # itrnum+=1
        # if itrnum<1000:
        # return Kepler(E2, M, ecc, itrnum=itrnum)
        # else:
        # return Kepler(E2+0.1, M, ecc, itrnum=1)


## Given true anomaly nu and parameters, function returns an 2-col array of modeled (Q,U)
# def v1andv2(nu, Gamma1, Gamma2, K1, K2, Omega, ecc):
# Omega = Omega/180. * np.pi
# v1 = Gamma1 + K1*(np.cos(Omega + nu) + ecc* np.cos(Omega))
# v2 = Gamma2 + K2*(np.cos(np.pi + Omega + nu) + ecc* np.cos(np.pi + Omega))
# return np.column_stack((v1,v2))

# Given true anomaly nu and parameters, function returns an 2-col array of modeled (Q,U)
def v1mod(nu, Gamma1, K1, Omega, ecc):
    Omega = Omega / 180. * np.pi
    v1 = Gamma1 + K1 * (np.cos(Omega + nu) + ecc * np.cos(Omega))
    return v1


# Given true anomaly nu and parameters, function returns an 2-col array of modeled (Q,U)
def v2mod(nu, Gamma2, K2, Omega, ecc):
    Omega = Omega / 180. * np.pi
    v2 = Gamma2 + K2 * (np.cos(np.pi + Omega + nu) + ecc * np.cos(np.pi + Omega))
    return v2


def nus1(hjds, P, T0, ecc):
    phis = (hjds - T0) / P - ((hjds - T0) / P).astype(int)
    phis[phis < 0] = phis[phis < 0] + 1.
    Ms = 2 * np.pi * phis
    Es = Kepler(np.pi, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    # nusdata[nusdata < 0] = nusdata[nusdata < 0] + 2*np.pi
    return nusdata


# Target function (returns differences between model array with parameter set p and data)
def chisqr(p):
    Gamma1 = p['Gamma1'].value
    # Gamma2 = p['Gamma2'].value
    K1 = p['K1'].value
    # if CompNum==2:
    # K2 = p['K2'].value
    Omega = p['Omega'].value
    ecc = p['ecc'].value
    T0 = p['T0'].value
    P = p['P'].value
    # ShiftSec = p['ShiftSec'].value
    v1 = v1mod(nus1(hjds1, P, T0, ecc), Gamma1, K1, Omega, ecc)
    # v2 = v2mod(nus1(hjds2, P, T0, ecc), Gamma1, K2, Omega, ecc)
    # v2s_shift = np.copy(v2s) + ShiftSec
    # return np.concatenate(((v1 - v1s)/errv1s,(v2 - v2s_shift)/errv2s))
    return (v1 - v1s) / errv1s


# Binning of RVs, bintime = in days
def Bin_RVs(HJDs1, RVs1, HJDs2, RVs2, errv1s, errv2s, bintime=BinTime, consterr=3.):
    HJDsbin1, RVsbin1, HJDsbin2, RVsbin2 = [], [], [], []
    BinInds1 = [-1]
    BinInds2 = [-1]
    NewBin = True
    for i in np.arange(len(HJDs1) - 1):
        # print NewBin
        if NewBin:
            hjd0 = HJDs1[i]
            NewBin = False
        # print NewBin, hjd0, HJDs1[i]
        if HJDs1[i + 1] - hjd0 > bintime:
            BinInds1.append(i)
            NewBin = True
    NewBin = True
    for i in np.arange(len(HJDs2) - 1):
        if NewBin:
            hjd0 = HJDs2[i]
            NewBin = False
        if HJDs2[i + 1] - hjd0 > bintime:
            BinInds2.append(i)
            NewBin = True
    BinInds1.append(-2)
    BinInds2.append(-2)
    # print BinInds1
    # print BinInds2,len(  HJDs1)
    HJDs1new = np.array([np.mean(HJDs1[BinInds1[i] + 1:BinInds1[i + 1] + 1]) for i in np.arange(len(BinInds1) - 1)])
    HJDs2new = np.array([np.mean(HJDs2[BinInds2[i] + 1:BinInds2[i + 1] + 1]) for i in np.arange(len(BinInds2) - 1)])
    RVs1new = np.array([np.mean(RVs1[BinInds1[i] + 1:BinInds1[i + 1] + 1]) for i in np.arange(len(BinInds1) - 1)])
    RVs2new = np.array([np.mean(RVs2[BinInds2[i] + 1:BinInds2[i + 1] + 1]) for i in np.arange(len(BinInds2) - 1)])
    errv1snew = np.array([max(.5, (consterr ** 2 + np.mean(errv1s[BinInds1[i] + 1:BinInds1[i + 1] + 1])) ** 0.5 / (
                BinInds1[i + 1] - BinInds1[i])) for i in np.arange(len(BinInds1) - 1)])
    errv2snew = np.array([max(.5, (consterr ** 2 + np.mean(errv2s[BinInds2[i] + 1:BinInds2[i + 1] + 1])) ** 0.5 / (
                BinInds2[i + 1] - BinInds2[i])) for i in np.arange(len(BinInds2) - 1)])
    # for i in np.arange(len(BinInds1)-1):
    ###print BinInds1[i]+1, BinInds1[i+1]+1
    # dasds
    return HJDs1new, RVs1new, HJDs2new, RVs2new, errv1snew, errv2snew


mini = lmfit.Minimizer(chisqr, params)
result = mini.minimize(method=MiniMethod)
print(lmfit.fit_report(result))
print(lmfit.fit_report)  # will print you a fit report from your model
Gamma1_res = result.params['Gamma1'].value
# Gamma2 = result.params['Gamma2'].value
K1_res = result.params['K1'].value
# if CompNum==2:
# K2 = result.params['K2'].value
Omega_res = result.params['Omega'].value
ecc_res = result.params['ecc'].value
P_res = result.params['P'].value
T0_res = result.params['T0'].value
# ShiftSec =  result.params['ShiftSec'].value

# v2s += ShiftSec

JDsplot = np.arange(min(hjds1) - 0.5 * P_res, max(hjds1) + 0.5 * P_res, .1)
# print 'hi'
phsplot = (JDsplot - T0_res) / P_res - ((JDsplot - T0_res) / P_res).astype(int)
# print('hi1'
Ms = 2 * np.pi * phsplot
# print('hi')
Es = Kepler(np.pi, Ms, ecc_res)
# print('hi2')
eccfac = np.sqrt((1 + ecc_res) / (1 - ecc_res))
nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
plt.errorbar(hjds1, v1s, yerr=errv1s, fmt='o', color='black')
plt.plot(JDsplot, v1mod(nusdata, Gamma1_res, K1_res, Omega_res, ecc_res), color='black', label='primary')
plt.xlabel('MJD')
plt.ylabel(r'RV [${\rm km}\,{\rm s}^{-1}$]')
plt.legend(loc=locleg)
# plt.savefig('Orbit_MJD.pdf', bbox_inches='tight')
plt.show()
# Create continous phase grid
phs = np.linspace(0., 1., num=1000)
Ms = 2 * np.pi * phs
Es = Kepler(np.pi, Ms, ecc_res)
eccfac = np.sqrt((1 + ecc_res) / (1 - ecc_res))
nus = 2. * np.arctan(eccfac * np.tan(0.5 * Es))

# Create data phase grid
phsdata = (hjds1 - T0_res) / P_res - ((hjds1 - T0_res) / P_res).astype(int)
phsdata[phsdata < 0] = phsdata[phsdata < 0] + 1.
phsdata0 = phsdata - 1.
phsdata2 = phsdata + 1.
phsdataall = np.concatenate((phsdata0, phsdata, phsdata2))
Ms = 2 * np.pi * phsdata
Es = Kepler(np.pi, Ms, ecc_res)
eccfac = np.sqrt((1 + ecc_res) / (1 - ecc_res))
nusdata = 2. * np.arctan(eccfac * np.tan(0.5 * Es))

phsdata1 = np.copy(phsdata)

fig, ax = plt.subplots(figsize=[6, 3])

plt.errorbar(phsdata, v1s, yerr=errv1s, fmt='o', color='red')
rms1 = (np.sum((v1mod(nusdata, Gamma1_res, K1_res, Omega_res, ecc_res) - v1s) ** 2) / len(v1s)) ** 0.5

plt.plot(phs, v1mod(nus, Gamma1_res, K1_res, Omega_res, ecc_res), color='red', label=r'Primary')
plt.plot([0., 1., ], [Gamma1_res, Gamma1_res], color='black')
plt.xlim(0., 1.)
plt.legend()
plt.ylabel(r'RV $[{\rm km}\,{\rm s}^{-1}]$')
plt.xlabel(r'phase')

# plt.savefig('Orbit2c.pdf', bbox_inches='tight')
plt.show()

print("RMS1 is ", rms1)
# print("RMS2 is ", rms2                                )
print("mean Error: ", np.mean(errv1s))

# print("mean Error: ", np.mean(errv2s)                 )
print("multiply errors1 by: ", rms1 / np.mean(errv1s))
# print("multiply errors2 by: ", rms2 / np.mean(errv2s) )


result.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
res = lmfit.minimize(chisqr, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20,
                     params=result.params, is_weighted=False, progress=False)

# result_emcee = mini.minimize(params=emcee_params)
print(res.var_names)
print(list(res.params.valuesdict().values()))
truths = [P, GAMMA, K1, OMEGA_rad, ECC, T0, 0]
emcee_plot = corner.corner(res.flatchain, labels=res.var_names,
                           truths=truths)
plt.show()

