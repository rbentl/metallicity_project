import subprocess

from matplotlib.backends.backend_pdf import PdfPages

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
from specutils import Spectrum1D,rvmeasure

from scipy import signal
import datetime

import sys

import gc
from astropy.io import fits
from astropy.time import Time
import glob


gc.enable()

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()



# load the BOSZ grid. Do this only ONCE!! It takes a long time and lots of memory
#g = load_grid('/u/rbentley/metallicity/spectra_fits/phoenix_t2500_6000_w20000_24000_R40000.h5')#phoenix_t2500_6000_w20000_24000_R40000.h5')#

starname = sys.argv[1]
specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
order = sys.argv[2]
radv = sys.argv[3]
teff = float(sys.argv[4])
logg = float(sys.argv[5])
mh = float(sys.argv[6])
snr = float(sys.argv[7])
q = 0
print starname,order,radv,teff,logg,mh,snr
omaxes = []
xparam = []
yparam = []

#file2 = glob.glob(specdir+starname+'_order'+order+'*.dat')
#starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom')
#waverange = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]

    
lkmax = None
maxparams = None
allvals = []
print starname,order,radv,teff,logg,mh,snr
#proc = fitting(starname,str(order),radv,str(teff),str(logg),str(mh),str(snr),g)

proc = subprocess.Popen(['python','fitting.py',starname,str(order),radv,str(teff),str(logg),str(mh),str(snr)])
#print proc
while True:
    if proc.poll() is not None:
        break
    #residual_mm = proc.returncode
    #print residual_mm
gc.collect()

print "done with order",order






    
'''
    for i in np.linspace(0.1, 2.1, 20):
        proc = subprocess.Popen(['python','fitting.py',starname,str(order),radv,str(teff*i),str(logg),str(mh),chi2max], stdout=subprocess.PIPE)
        while True:
            output = proc.stdout.readline()
            print output
            if 'ln(ev)' in output:
                likelihood = output.split('  ')
                lk, lkerr = float(likelihood[1]), float(likelihood[5])
                #chi2 = float(likelihood[1])
                print lk, lkerr
            elif proc.poll() is not None:
                break
        allvals += [(lk,lkerr,[radv,teff*i,logg,mh])]
        xparam += [teff*i]
        yparam += [lk]
        print "done", lk, lkerr, [radv,teff*i,logg,mh]
        print lkmax, lk
        if lkmax is None:
            lkmax = (lk, lkerr)
            maxparams = [radv,teff*i,logg,mh]
        elif lk > lkmax[0]:
            lkmax = (lk, lkerr)
            maxparams = [radv,teff*i,logg,mh]
            print "new max at",radv,teff*i,logg,mh
        
        if chi2max is None:
            chi2max = (chi2)
            maxparams = [radv,teff,logg*i,mh]
        
        elif chi2 > chi2max:
            chi2max = chi2
            maxparams = [radv,teff,logg*i,mh]
            print "new max at",radv,teff,logg*i,mh
    
        
'''
'''
    print lk, "fully done with order", order
    print allvals
    omaxes += [(lkmax,maxparams,order)]

    print omaxes
    plt.plot(xparam,yparam)
    plt.text(0.5,0.95,'max likelihood is '+str(lkmax[0])+' at '+str(maxparams[1]))
    plt.xlabel('Teff')
    plt.ylabel('Likelihood')
    plt.savefig('Teff_likelihood_'+starname+'_order'+order+'.pdf')
'''
'''
for i in np.linspace(0.9, 1.1, 4):
    for j in np.linspace(0.9, 1.1, 4):
        for k in np.linspace(0.9, 1.1, 4):
            for l in np.linspace(0.9, 1.1, 4):
                proc = subprocess.Popen(['python','fitting.py',starname,order,radv*l,str(teff*i),str(logg*j),str(mh*k)], stdout=subprocess.PIPE)
                while True:
                    output = proc.stdout.readline()
                    if 'ln(ev)' in output:
                        likelihood = output.split('  ')
                        lk, lkerr = float(likelihood[1]), float(likelihood[5])
                        print lk, lkerr
                    elif proc.poll() is not None:
                        break
                allvals += [(lk,lkerr,[radv*l,teff*i,logg*j,mh*k])]
                print "done", lk, lkerr, [radv*l,teff*i,logg*j,mh*k]
    
                if lkmax is None:
                    lkmax = (lk, lkerr)
                    maxparams = [radv*l,teff*i,logg*j,mh*k]
                elif lk > lkmax[0]:
                    lkmax = (lk, lkerr)                    maxparams = [radv*l,teff*i,logg*j,mh*k]
                    print "new max at",radv*l,teff*i,logg*j,mh*k
'''

