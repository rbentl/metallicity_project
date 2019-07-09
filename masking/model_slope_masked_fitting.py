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

unmasked_median_fits = {"teff_0":4026.56,
                        "logg_0":1.653,
                        "mh_0":0.447044,
                        "alpha_0":0.020058,
                        "vrot_1":0.928,
                        "vrad_2":-48.4008,
                        "R_3":24000.}


cutoff = sys.argv[1]

cutoff_str = str(cutoff)
            
#sigma_cut_val = float(sys.argv[1])
#print "Cutting at ",sigma_cut_val

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

print len(starspectrum35.flux.value)

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
slopes = mt.find_slopes(model, 'mh', np.arange(-0.5,0.8,0.1))
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1, 1, 1)
slopeax=plt.twinx()

print slopes, np.median(slopes), np.std(slopes)

residual_flux = mt.calc_residuals(f,starspectrum35.flux.value)

plt.figure(figsize=(12,10))

#plt.plot(masked_data_sl_w,masked_data_sl_f)

slope_masked_flux, slope_masked_wavelength, slope_masked_uncert = mt.slope_masked_data(starspectrum35.wavelength.value,starspectrum35.flux.value,starspectrum35.uncertainty.value,slopes,cutoff)
print type(slope_masked_flux)

masked_data_slope = Spectrum1D.from_array(dispersion=slope_masked_wavelength, flux=slope_masked_flux, dispersion_unit=u.angstrom, uncertainty=slope_masked_uncert)

interp_slope = Interpolate(masked_data_slope)
convolve_slope = InstrumentConvolveGrating.from_grid(g,R=24000)
rot_slope = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
norm_slope = Normalize(masked_data_slope,2)
like_slope = Chi2Likelihood(masked_data_slope)
                         
model = g | rot_slope |DopplerShift(vrad=0.0)| convolve_slope | interp_slope | norm_slope
masked_model_slope = model | like_slope
tw,tf = model()

masked_model_slope



#Fits S_lambda masked model
gc_result_masked_slope = mt.run_multinest_fit(masked_model_slope)



gc_result_masked_slope.to_hdf("/u/rbentley/metallicity/spectra_fits/masked_fit_results/slope_cut_"+cutoff_str+"_NGC6791_J19205+3748282_order35.h5")

print "chi squared val ", like_slope


#Makes plots of S_lambda masked model

#print gc_result
print gc_result_masked_slope

for a in gc_result_masked_slope.median.keys():
    setattr(model,a,gc_result_masked_slope.median[a])

slope_w,slope_f = model()

print len(w), len(slope_w)

plt.tick_params(axis='both', which='major', labelsize=11)
plt.plot(starspectrum35.wavelength.value/(unmasked_median_fits['vrad_2']/3e5+1.0), starspectrum35.flux.value+0.5, label="Data")

plt.plot(w/(unmasked_median_fits['vrad_2']/3e5+1.0),f,label="Unmasked model with best fit unmasked values")

plt.plot(slope_w/(gc_result_masked_slope.median['vrad_2']/3e5+1.0),slope_f-0.5,label="Masked model with best fit masked values")

plt.plot(slope_w/(gc_result_masked_slope.median['vrad_2']/3e5+1.0),masked_data_slope.flux.value-slope_f,label='Masked Model-Masked Data Residuals')

slopeax.plot(w/(unmasked_median_fits['vrad_2']/3e5+1.0), slopes,label="Slopes")

slopeax.set_ylim(np.amin(slopes)*3.,np.amax(slopes)*3.)

slopeax.legend(loc='lower right', fontsize=11)

plt.xlabel("Wavelength (angstroms)", size=12)
plt.ylabel("Flux", size=12)
plt.ylim(-0.3,1.7)
plt.title("Order 35 Observed and Model fluxes (> "+cutoff_str+" slope masked)", size=15)
plt.legend(loc='upper right', fontsize=11)
plt.tick_params(axis='both', which='major', labelsize=11)

plt.show()

axpresent = fig.add_subplot(1, 1, 1)

axpresent.plot(starspectrum35.wavelength.value/(unmasked_median_fits['vrad_2']/3e5+1.0), starspectrum35.flux.value+1.5, label="Data")

axpresent.plot(slope_w/(gc_result_masked_slope.median['vrad_2']/3e5+1.0),slope_f,label="Masked model with best fit masked values")

axpresent.plot(slope_w/(gc_result_masked_slope.median['vrad_2']/3e5+1.0),masked_data_slope.flux.value-slope_f+0.5,label='Masked Model-Masked Data Residuals')

plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])
#plt.plot(sl_w,unmasked_mod_masked_data_removed_flux-sl_f,label="Umasked model-masked model residuals")

axpresent.set_xlabel("Wavelength (angstroms)", size=12)
axpresent.set_ylabel("Flux", size=12)
axpresent.set_ylim(-0.3,1.2)
axpresent.set_title("Order 35 Observed and Model fluxes (unmasked)", size=15)
axpresent.legend(loc='upper right', fontsize=11)



fit_results_file = open("/u/ghezgroup/data/metallicity/nirspec/spectra_fits/pdfs/slope_masked_fit_results_output.lis","a+")
fit_results_file.write("Fitting results with slopes with slope < "+cutoff_str+" removed\n")
fit_results_file.write(str(gc_result_masked_slope)+"\n\n\n")
fit_results_file.close()



#multipage("/u/ghezgroup/data/metallicity/nirspec/spectra_fits/pdfs/NGC6791_J19205+3748282_order35_median_slope_cut.pdf")
