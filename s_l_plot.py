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

unmasked_median_fits = {"teff_0":3891.,
                        "logg_0":0.104,
                        "mh_0":0.0023,
                        "alpha_0":0.305,
                        "vrot_1":0.928,
                        "vrad_2":140.6,
                        "R_3":24000.}

sl_masked_median_fits = {"teff_0":3891.,
                        "logg_0":0.104,
                        "mh_0":0.0023,
                        "alpha_0":0.305,
                        "vrot_1":0.928,
                        "vrad_2":140.6,
                        "R_3":24000.}
            
sl_cut_val = float(sys.argv[1])
print "Cutting at ",sl_cut_val

specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
testspec_list = specdir+'NGC6791_J19205+3748282_order35*.dat'
testspec_path = glob.glob(testspec_list)

snr = 50.

starspectrumall = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='micron')
    
#waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]
waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]
starspectrum35 = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='Angstrom',
                                                 wave_range=waverange)
starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

print testspec_path

g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2000_6000_w21500_22220_R40000_o35.h5') #for order 35
#g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w22350_22900_R40000_o34.h5') #for order 34

w,f = g()

testspec_w = np.linspace(w[0],w[-1],np.amax(w)-np.amin(w))
testspec_f = np.ones(len(testspec_w))
testspec_u = np.ones(len(testspec_w))*0.001
testspec = SKSpectrum1D.from_array(wavelength=testspec_w*u.angstrom, flux=testspec_f*u.Unit('erg/s/cm^2/angstrom'), uncertainty=testspec_u*u.Unit('erg/s/cm^2/angstrom'))


interp1 = Interpolate(starspectrum35)
convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
norm1 = Normalize(starspectrum35,2)

# concatenate the spectral grid (which will have the stellar parameters) with other 
# model components that you want to fit
model = g | rot1 |DopplerShift(vrad=0.0)| convolve1 | interp1 | norm1

# add likelihood parts
like1 = Chi2Likelihood(starspectrum35)
#like1_l1 = SpectralL1Likelihood(spectrum)

fit_model = model | like1


#print fit_model



#This is the fit itself. gc_result is a set of parameters from the best MultiNest fit to the data. 
#This cell takes time to evaluate.
'''
gc_result = mt.run_multinest_fit(fit_model)



for a in gc_result.median.keys():
    setattr(model,a,gc_result.median[a])
'''

for a in unmasked_median_fits.keys():
    setattr(model,a,unmasked_median_fits[a])
    
w, f = model()
print model

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1, 1, 1)

residual_flux = mt.calc_residuals(f,starspectrum35.flux.value)

#sl_hold = sl_mh

sl_mh = mt.s_lambda(model,'mh',model.mh_0.value,0.1)

logsl = np.log10(map(abs, sl_mh))

sl_ax = ax.twinx()

ax.plot(starspectrum35.wavelength.value/(unmasked_median_fits['vrad_2']/3e5+1.0), starspectrum35.flux.value, label="Data")

ax.plot(w/(unmasked_median_fits['vrad_2']/3e5+1.0),f,label="Unmasked model with best fit unmasked values")

sl_ax.plot(w/(unmasked_median_fits['vrad_2']/3e5+1.0),logsl,label="Log($S_{\lambda}$)", color="green")

ax.plot(w/(unmasked_median_fits['vrad_2']/3e5+1.0),logsl+100.,label="Log($S_{\lambda}$)", color="green")

ax.set_xlabel("Wavelength (angstroms)", size=12)
ax.set_ylabel("Flux")
sl_ax.set_ylabel("Log($S_{\lambda}$)", size=12)
ax.set_ylim(-0.3,1.3)

sl_ax.set_ylim(-4,6.)
ax.set_title("Order 35 Observed and Model fluxes, with sensitivity function", size=15)
ax.legend(loc='upper right', fontsize=11)

plt.tick_params(axis='both', which='major', labelsize=11)


plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.25,size=12)
plt.show()
