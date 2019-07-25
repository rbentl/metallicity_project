import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import units as u
from starkit.fitkit import likelihoods
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from specutils import read_fits_file,plotlines
import shutil, logging, datetime
import os,scipy
from scipy import signal
from specutils import Spectrum1D,rvmeasure
import subprocess
import gc
from astropy.io import fits
from astropy.time import Time
import glob

from matplotlib.backends.backend_pdf import PdfPages

try:
    import MySQLdb as mdb
except:
    import pymysql as mdb

def retrive(starname, param):
    teff = []
    tefferr = []
    order = []
    passwd = 'w3p6QXyYx'
    con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbwrite',passwd=passwd,db='gcg')
    cur = con.cursor()
    sql_query = "SELECT "+param+", "+param+"_err, order_ FROM starkit_nirspec WHERE name ='"+starname+"'"
    cur.execute(sql_query)
    con.commit()
    result = cur.fetchall()
    for row in result:
        teff += [row[0]]
        tefferr += [row[1]]
        order += [row[2]]
    #print teff, starname
    cur.close()
    con.close()
    return (teff,tefferr,order)


def model_data(starname,order):
    models = pd.read_table('/u/ghezgroup/data/metallicity/nirspec/spectra_fits/ref_fits/model_plotvals_'+starname+'_order'+order+'.tsv', names = ('wave','bestfit','ref'), sep='\t')
    print models
    mw = models['wave'].tolist()
    mf = models['bestfit'].tolist()
    rf = models['ref'].tolist()
    print mw
    data = pd.read_table('/u/ghezgroup/data/metallicity/nirspec/spectra_fits/ref_fits/data_plotvals_'+starname+'_order'+order+'.tsv', names = ('wave','flux'), sep='\t')
    dw = data['wave'].tolist()
    df = data['flux'].tolist()
    resampled_mw = signal.resample(mw, len(dw))
    resampled_mf = signal.resample(mf, len(dw))
    resampled_rf = signal.resample(rf, len(dw))
    res_f = df - resampled_mf
    return (resampled_mw,res_f)


starnamelist = ['M5 J15190+0208','M71_J19534827+1848021','NGC6791_J19205+3748282','NGC6791_J19213390+3750202','NGC6819_J19411+4010517','NGC6819_J19413439+4017482','TYC 3544']
#starnamelist = ['M5 J15190+0208','M71_J19534827+1848021','TYC 3544','NGC6819_J19411+4010517','NGC6819_J19413439+4017482','NGC6791_J19213390+3750202','NGC6791_J19205+3748282']
mhlist = [-1.26,-0.7,-0.02,0.05,0.11,0.33,0.44]
orders = range(34,35)
tefflist = [3989,4060.3,4082,3901.9,4118.5,4196.4,5556]
tefffiterrs = [133.9,111.5,66.5,70.6,81.9,79.5,69.3]

mhlist = [-1.26,-0.7,0.3,0.4,0.05,0.11,0.3]
mhfiterrs = [0.1,0.08,0.021,0.04,0.053,0.051,0.02]

logglist = [0.72,1.08,1.63,1.33,1.56,1.75,0.0] #TYC 3544 logg of 0 is a placeholder until a value is found
loggfiterrs = [0.101,0.101,0.08,0.08,0.101,0.101,0.101] #errors of 0.101 are estimates based on Mesazaros et al. 2013 discussion

alphalist = [0.12,0.23,0.07,0.14,0.03,0.03,-0.02]
alphafiterrs = [0.05,0.05,0.011,0.011,0.05,0.05,0.01] #errors of 0.101 are estimates based on Mesazaros et al. 2013 discussion

colors = [('b',32),('c',33),('g',34),('y',35),('r',36),('m',37)]
shapes = ['o','D','v','^','<','>','x']


#plt.ylabel('Residual Flux')
#plt.xlabel('Wavelength (Angstroms)')
'''
h = 0.0
for o in range(len(orders)):
    for s in range(len(starnamelist)):
        resampled_mw, res_f = model_data(starnamelist[s],str(orders[o]))
        plt.plot(resampled_mw,h+res_f,label='Residuals '+starnamelist[s]+' ([M/H] = '+str(mhlist[s])+')')
        plt.axhline(y=h+0.03, color='r', linestyle='-')
        plt.axhline(y=h+0.0, color='g', linestyle='-')
        plt.axhline(y=h-0.03, color='r', linestyle='-')
        h += 0.2

        
'''

plt.figure(figsize=(15,7))



param = 'alpha'
for s in range(len(starnamelist)):
    alpha,alphaerr,order = retrive(starnamelist[s], param)
    refalpha = alphalist[s]
    for i in range(len(alpha)):
        quaderr= np.sqrt(alphaerr[i]**2+alphafiterrs[s]**2)
        for j in range(len(colors)):
          if order[i] == colors[j][1]:
             pltcolor = colors[j][0]
        plt.errorbar(refalpha,(alpha[i]-refalpha)/quaderr,yerr=1.,xerr=alphafiterrs[s], fmt=pltcolor+shapes[s])

plt.ylabel('Number of deviations seperation')
plt.xlabel('APOGEE Alpha Abundance')
plt.title('Fitted vs APOGEE Alpha')
#plt.xlim(3850,4250)
plt.axhline(y=3., linestyle='dashed', color='r')
plt.axhline(y=0., linestyle='dashed', color='g')
plt.axhline(y=-3., linestyle='dashed', color='r')
#plt.xlim(3250,4750)
#plt.ylim(-750,750)
plt.minorticks_on()

'''
param = 'logg'
for s in range(len(starnamelist)):
    logg,loggerr,order = retrive(starnamelist[s], param)
    reflogg = logglist[s]
    for i in range(len(logg)):
        quaderr= np.sqrt(loggerr[i]**2+loggfiterrs[s]**2)
        for j in range(len(colors)):
          if order[i] == colors[j][1]:
             pltcolor = colors[j][0]
        plt.errorbar(reflogg,(logg[i]-reflogg)/quaderr,yerr=1.,xerr=loggfiterrs[s], fmt=pltcolor+shapes[s])

plt.ylabel('Number of deviations seperation')
plt.xlabel('APOGEE Log g')
plt.title('Fitted vs APOGEE Log g')
#plt.xlim(3850,4250)
plt.axhline(y=3., linestyle='dashed', color='r')
plt.axhline(y=0., linestyle='dashed', color='g')
plt.axhline(y=-3., linestyle='dashed', color='r')
#plt.xlim(3250,4750)
#plt.ylim(-750,750)
plt.minorticks_on()
'''
'''
param = 'mh'
for s in range(len(starnamelist)):
    mh,mherr,order = retrive(starnamelist[s], param)
    refmh = mhlist[s]
    for i in range(len(mh)):
        quaderr= np.sqrt(mherr[i]**2+mhfiterrs[s]**2)
        for j in range(len(colors)):
          if order[i] == colors[j][1]:
             pltcolor = colors[j][0]
        plt.errorbar(refmh,(mh[i]-refmh)/quaderr,yerr=1.,xerr=mhfiterrs[s], fmt=pltcolor+shapes[s])

plt.ylabel('Number of deviations seperation')
plt.xlabel('APOGEE [M/H]')
plt.title('Fitted vs APOGEE [M/H]')
#plt.xlim(3850,4250)
plt.axhline(y=3., linestyle='dashed', color='r')
plt.axhline(y=0., linestyle='dashed', color='g')
plt.axhline(y=-3., linestyle='dashed', color='r')
#plt.xlim(3250,4750)
#plt.ylim(-750,750)
plt.minorticks_on()
'''
'''
param = 'teff'
for s in range(len(starnamelist)):
    teff,tefferr,order = retrive(starnamelist[s],param)
    refteff = tefflist[s]
    for i in range(len(teff)):
        quaderr= np.sqrt(tefferr[i]**2+tefffiterrs[s]**2)
        for j in range(len(colors)):
          if order[i] == colors[j][1]:
             pltcolor = colors[j][0]
        plt.errorbar(refteff,(teff[i]-refteff)/quaderr,yerr=1.,xerr=tefffiterrs[s], fmt=pltcolor+shapes[s])

plt.ylabel('Number of deviations seperation')
plt.xlabel('APOGEE Teff')
plt.title('Fitted vs APOGEE Teff')
#plt.xlim(3850,4250)
plt.axhline(y=3., linestyle='dashed', color='r')
plt.axhline(y=-3., linestyle='dashed', color='r')
#plt.xlim(3250,4750)
#plt.ylim(-750,750)
plt.minorticks_on()
'''



#plt.ylim(-0.6,1.4)
#plt.xlim(21775,22200)
#plt.legend(loc = (0.74,0.4))
#plt.minorticks_on()
#plt.title('Order 35 residuals')
#plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.3,size=11,highlight=['Sc','Fe','Ti'])

plt.show()
    
    

