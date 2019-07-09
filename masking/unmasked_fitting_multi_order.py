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
from specutils import read_fits_file,plotlines,combine_spectra
import numpy as np
import os,scipy
from specutils import Spectrum1D,rvmeasure
import datetime,glob
import model_tester_updated as mt
from matplotlib.backends.backend_pdf import PdfPages

import sys


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


#This cell loads the data and the model grid. It then concatenates the grid to the model parameters such as 
#v_rad, v_rot. All data is used here, with no masking

apogee_vals = {"teff_0":4026.56,
                        "logg_0":1.653,
                        "mh_0":0.447044,
                        "alpha_0":0.020058,
                        "vrot_1":0.928,
                        "vrad_2":-48.4008,
                        "R_3":24000.}



starname = 'NGC6791_J19205+3748282'

order_range = range(34,37)

specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'

starspectrum_flux = None
starspectrum_uncert = None
starspectrum_wavelength = None
waveranges = []

allorders_path = []

for order in order_range:
    testspec_list = specdir+starname+'_order'+str(order)+'*.dat'

    testspec_path = glob.glob(testspec_list)

    snr = mt.get_snr(starname, str(order))

    starspectrumall = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='micron')
    
    waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]


    
    single_order_spec = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='Angstrom',
                                                          wave_range=waverange)

    single_order_spec.uncertainty = (np.zeros(len(single_order_spec.flux.value))+1.0/np.float(snr))*single_order_spec.flux.unit


    if starspectrum_flux is None:

        starspectrum_flux = single_order_spec.flux.value

        starspectrum_wavelength = single_order_spec.wavelength.value
        
        starspectrum_uncert = single_order_spec.uncertainty.value
        
    else:
        
        starspectrum_uncert = np.concatenate((starspectrum_uncert, single_order_spec.uncertainty.value[::-1]))

        starspectrum_flux = np.concatenate((starspectrum_flux, single_order_spec.flux.value[::-1]))
        
        starspectrum_wavelength = np.concatenate((starspectrum_wavelength, single_order_spec.wavelength.value[::-1]))
    
#print allorders_path
#starspectrum = read_fits_file.read_nirspec_dat(allorders_path,desired_wavelength_units='Angstrom',
#                                                 wave_range=waveranges)        
starspectrum = Spectrum1D.from_array(dispersion=starspectrum_wavelength, flux=starspectrum_flux, dispersion_unit=u.angstrom, uncertainty=starspectrum_uncert)


g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

w,f = g()
print len(starspectrum.flux.value)
interp1 = Interpolate(starspectrum)
convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
norm1 = Normalize(starspectrum,2)

# concatenate the spectral grid (which will have the stellar parameters) with other 
# model components that you want to fit
model = g | rot1 |DopplerShift(vrad=0.0)| convolve1 | interp1 | norm1

# add likelihood parts
like1 = Chi2Likelihood(starspectrum)
#like1_l1 = SpectralL1Likelihood(spectrum)

fit_model = model | like1

w, f = model()
print model

fig = plt.figure(figsize=(12,10))
fig2 = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1, 1, 1)


gc_result = mt.run_multinest_fit_rv_constrained(fit_model,apogee_vals['vrad_2'],0.11)


gc_result.to_hdf("/u/rbentley/metallicity/spectra_fits/masked_fit_results/unmasked_rv_constrained_"+starname+"_orders34-36.h5")

print "chi squared val ", like1

sigmas = gc_result.calculate_sigmas(1)

#Makes plots of S_lambda masked model

#print gc_result
print gc_result
for a in gc_result.median.keys():
    setattr(model,a,gc_result.median[a])

w,f = model()
gc_result.plot_triangle(parameters=['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2'])
plt.figure(figsize=(15,7))
plt.plot(starspectrum.wavelength.value/(gc_result.median['vrad_2']/3e5+1.0),starspectrum.flux)
plt.plot(w/(gc_result.median['vrad_2']/3e5+1.0),f)
plotlines.oplotlines(angstrom=True)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux')
plt.show()
