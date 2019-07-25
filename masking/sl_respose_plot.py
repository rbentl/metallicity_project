import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
from astropy import units as u
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from starkit.fix_spectrum1d import SKSpectrum1D
from specutils import read_fits_file,plotlines
import numpy as np
import os,scipy
from specutils import Spectrum1D,rvmeasure
import datetime,glob
import model_tester_updated as mt
from matplotlib.backends.backend_pdf import PdfPages
import operator
import sys

def getKey(item):
    return item[0]

starname = 'NGC6791_J19205+3748282'

order = str(sys.argv[1])

specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
testspec_list = specdir+starname+'_order'+order+'*.dat'
testspec_path = glob.glob(testspec_list)

snr = mt.get_snr(starname, order)

print snr

starspectrumall = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='micron')

waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]
starspectrum35 = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='Angstrom',
                                                 wave_range=waverange)
starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit





h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/order36/*.h5')

sl_val = []
mask_len = []
vrad = []

if float(order) == 36.:
    g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w21000_21600_R40000_o36.h5') #for order 36

elif float(order) == 35.:
    g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2000_6000_w21500_22220_R40000_o35.h5') #for order 35

elif float(order) == 34.:
    g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w22350_22900_R40000_o34.h5') #for order 34

else:
    print "Need to code in that order"
    exit()
    
    
interp1 = Interpolate(starspectrum35)
convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
norm1 = Normalize(starspectrum35,2)

# concatenate the spectral grid (which will have the stellar parameters) with other 
# model components that you want to fit
model = g | rot1 |DopplerShift(vrad=0.0)| convolve1 | interp1 | norm1

cut_lis = []
for filename in h5_files_us:
    cut_lis += [(float(filename.split('_')[5]),filename)]

cut_lis = sorted(cut_lis,key = getKey)
print cut_lis
h5_files = [i[1] for i in cut_lis]

    
for filename in h5_files:



    gc_result = MultiNestResult.from_hdf5(filename)


    for a in gc_result.median.keys():
        setattr(model,a,gc_result.median[a])

    sl_mh = mt.s_lambda(model,'mh',model.mh_0.value,0.1)

    mask_sl_f = []
    mask_sl_w = []

    
    data_sl_f = []

    abs_sl_mh = []

    for i in range(len(sl_mh)):
        abs_sl_mh += [np.abs(sl_mh[i])]
        if abs(sl_mh[i]) < float(filename.split('_')[5]):
            mask_sl_f += [starspectrum35.flux.value[i]]
            mask_sl_w += [starspectrum35.wavelength.value[i]]
        else:
            data_sl_f += [starspectrum35.flux.value[i]]


    w,f  = model()
    '''
    plt.figure(figsize=(20,6))

    specax = plt.subplot()

    slax = specax.twinx()
    
    specax.plot(starspectrum35.wavelength.value, starspectrum35.flux.value, color='blue')

    specax.plot(w, f, color='green')

    specax.plot(mask_sl_w, mask_sl_f, 'r.')

    #slax.semilogy(w, sl_mh, color='0.5')

    slax.set_title(filename.split('_')[5] + '     '+str(len(mask_sl_f))+'     '+str(gc_result.median['vrad_2']))
    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])
    #sl_val += [float(filename.split('_')[5])]
    #mask_len += [len(mask_sl_f)]
    #vrad += [gc_result.median['vrad_2']]
    '''
    sl_val += [(float(filename.split('_')[5]),len(mask_sl_f),gc_result.median['vrad_2'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],w,f)]
 
    print gc_result
    print float(filename.split('_')[5]),len(mask_sl_f),gc_result.median['vrad_2']


f= plt.figure(figsize=(12,11))
rvax  = f.add_subplot(5,1,1)
loggax  = f.add_subplot(5,1,2,sharex=rvax)
mhax  = f.add_subplot(5,1,3)
alphaax = f.add_subplot(5,1,4)
lenax = f.add_subplot(5,1,5)

    
sl_val = sorted(sl_val)
#ax = plt.subplot()
    
rvax.plot([i[0] for i in sl_val],[i[2] for i in sl_val], color='red')
rvax.set_xscale('log')
rvax.set_ylabel('Radial Velocity')

loggax.plot([i[0] for i in sl_val],[i[3] for i in sl_val], color='blue')
loggax.set_xscale('log')
loggax.set_ylabel('Log g')

mhax.plot([i[0] for i in sl_val],[i[4] for i in sl_val], color='green')
mhax.set_xscale('log')
mhax.set_ylabel('[M/H]')

alphaax.plot([i[0] for i in sl_val],[i[5] for i in sl_val], color='yellow')
alphaax.set_xscale('log')
alphaax.set_ylabel('$alpha$')

lenax.plot([i[0] for i in sl_val],[i[1] for i in sl_val], color='black')
lenax.set_xscale('log')
lenax.set_xlabel('$S_{\lambda}$ cutoff')
lenax.set_ylabel('# points masked')


#ax2 = ax.twinx()

#ax2.plot([i[0] for i in sl_val],[i[1] for i in sl_val], 'bo')

#
plt.show()
