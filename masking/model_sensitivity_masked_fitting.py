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

apogee_vals = {"teff_0":4026.56,
                        "logg_0":1.653,
                        "mh_0":0.447044,
                        "alpha_0":0.020058,
                        "vrot_1":0.928,
                        "vrad_2":-48.4008,
                        "R_3":24000.}

            
sl_cut_val = float(sys.argv[1])
print "Cutting at ",sl_cut_val


starname = 'NGC6791_J19205+3748282'

order = str(36)

specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
testspec_list = specdir+starname+'_order'+order+'*.dat'
testspec_path = glob.glob(testspec_list)

snr = mt.get_snr(starname, order)

starspectrumall = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='micron')
    
#waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]
waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]
starspectrum35 = read_fits_file.read_nirspec_dat(testspec_path,desired_wavelength_units='Angstrom',
                                                 wave_range=waverange)
starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

print testspec_path

#g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2000_6000_w21500_22220_R40000_o35.h5') #for order 35
#g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w22350_22900_R40000_o34.h5') #for order 34
g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w21000_21600_R40000_o36.h5')

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

for a in apogee_vals.keys():
    setattr(model,a,apogee_vals[a])
    
w, f = model()
print model

fig = plt.figure(figsize=(12,10))
fig2 = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1, 1, 1)

residual_flux = mt.calc_residuals(f,starspectrum35.flux.value)

#sl_hold = sl_mh

sl_mh = mt.s_lambda(model,'mh',model.mh_0.value,0.1)

sl_teff = mt.s_lambda(model,'teff',model.teff_0.value,100)

R = mt.r_val_polynomial(model)



#Makes mask based on S_lambda, and concatenates as before

sl_mask_indices = []
mask_sl_f = []
mask_sl_w = []

unmasked_mod_masked_data_removed_flux = []

for i in range(len(sl_mh)):
    if abs(sl_mh[i])<float(sys.argv[1]):
        mask_sl_f += [starspectrum35.flux.value[i]]
        mask_sl_w += [starspectrum35.wavelength.value[i]]
        unmasked_mod_masked_data_removed_flux += [f[i]]

        sl_mask_indices += [i]

mask_sl_flux = np.array([starspectrum35.flux.value[i] for i in sl_mask_indices])

mask_sl_wavelength = np.array([starspectrum35.wavelength.value[i] for i in sl_mask_indices])


ax.plot(starspectrum35.wavelength.value,sl_mh,'b.')
ax.plot(mask_sl_w,mask_sl_f,'r.')




masked_data_sl_f = np.delete(starspectrum35.flux.value,sl_mask_indices)
masked_data_sl_w = np.delete(starspectrum35.wavelength.value,sl_mask_indices)
masked_data_sl_u = np.delete(starspectrum35.uncertainty.value,sl_mask_indices)

plt.figure(figsize=(12,10))

#plt.plot(masked_data_sl_w,masked_data_sl_f)


#masked_data_sl = SKSpectrum1D.from_array(wavelength=masked_data_sl_w*u.angstrom, flux=masked_data_sl_f*u.Unit('erg/s/cm^2/angstrom'), uncertainty=masked_data_sl_f*u.Unit('erg/s/cm^2/angstrom'))

masked_data_sl = Spectrum1D.from_array(dispersion=masked_data_sl_w, flux=masked_data_sl_f, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u) #

interp_sl = Interpolate(masked_data_sl)
convolve_sl = InstrumentConvolveGrating.from_grid(g,R=24000)
rot_sl = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
norm_sl = Normalize(masked_data_sl,2)
like_sl = Chi2Likelihood(masked_data_sl)
                         
model = g | rot_sl |DopplerShift(vrad=0.0)| convolve_sl | interp_sl | norm_sl
masked_model_sl = model | like_sl
tw,tf = model()

masked_model_sl

#Fits S_lambda masked model
print type(mask_sl_flux), type(masked_data_sl.wavelength.value)
gc_result_masked_sl = mt.run_multinest_fit(masked_model_sl)

cutoffstr = str(sys.argv[1])

gc_result_masked_sl.to_hdf("/u/rbentley/metallicity/spectra_fits/masked_fit_results/order36/sensitivity_cut_"+cutoffstr+"_"+starname+"_order"+order+".h5")

print "chi squared val ", like_sl

sigma_sl = gc_result_masked_sl.calculate_sigmas(1)

#Makes plots of S_lambda masked model

#print gc_result
print gc_result_masked_sl
for a in gc_result_masked_sl.median.keys():
    setattr(model,a,gc_result_masked_sl.median[a])

sl_w,sl_f = model()

#for a in gc_result.median.keys():
#    setattr(model,a,gc_result.median[a])

sl_unmasked_w,sl_unmasked_f = model()

plt.tick_params(axis='both', which='major', labelsize=11)

plt.plot(w/(apogee_vals['vrad_2']/3e5+1.0), starspectrum35.flux.value, label="Data")
'''
plt.plot(w/(unmasked_median_fits['vrad_2']/3e5+1.0),f,label="Unmasked model with best fit unmasked values")

plt.plot(sl_w/(gc_result_masked_sl.median['vrad_2']/3e5+1.0),sl_f-0.5,label="Masked model with best fit masked values")

plt.plot(sl_w/(gc_result_masked_sl.median['vrad_2']/3e5+1.0),masked_data_sl.flux.value-sl_f,label='Masked Model-Masked Data Residuals')

'''
plt.plot(w/(apogee_vals['vrad_2']/3e5+1.0),f,label="Best Fit Model (No mask)")

#plt.plot(sl_w/(gc_result_masked_sl.median['vrad_2']/3e5+1.0),sl_f-0.5,label="Masked model with best fit masked values")

#plt.plot(w/(gc_result_masked_sl.median['vrad_2']/3e5+1.0),starspectrum35.flux.value-f,label='Masked Model-Masked Data Residuals')

plt.plot(w/(apogee_vals['vrad_2']/3e5+1.0),starspectrum35.flux.value-f,label='Data - Model Residuals')

plt.xlabel("Wavelength (angstroms)", size=12)
plt.ylabel("Flux", size=12)
plt.ylim(-0.3,1.2)
#plt.title("Order 35 Observed and Model fluxes (regions with $S_{\lambda}$ < "+str(sys.argv[1])+" masked)", size=15)

plt.title("Order 36 Observed and Model fluxes (No mask)", size=15)

plt.legend(loc='lower right', fontsize=11)
plt.tick_params(axis='both', which='major', labelsize=11)


axpresent = fig2.add_subplot(1, 1, 1)

axpresent.plot(starspectrum35.wavelength.value/(apogee_vals['vrad_2']/3e5+1.0), starspectrum35.flux.value+1.5, label="Data")

axpresent.plot(sl_w/(gc_result_masked_sl.median['vrad_2']/3e5+1.0),sl_f,label="Masked model with best fit masked values")

axpresent.plot(sl_w/(gc_result_masked_sl.median['vrad_2']/3e5+1.0),masked_data_sl.flux.value-sl_f+0.5,label='Masked Model-Masked Data Residuals')

plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])
#plt.plot(sl_w,unmasked_mod_masked_data_removed_flux-sl_f,label="Umasked model-masked model residuals")

axpresent.set_xlabel("Wavelength (angstroms)", size=12)
axpresent.set_ylabel("Flux", size=12)
axpresent.set_ylim(-0.3,1.2)
axpresent.set_title("Order 36 Observed and Model fluxes (unmasked)", size=15)
axpresent.legend(loc='upper right', fontsize=11)



fit_results_file = open("/u/ghezgroup/data/metallicity/nirspec/spectra_fits/pdfs/sl_masked_fit_results_output.lis","a+")
fit_results_file.write("Fitting results with S_l < "+str(sys.argv[1])+" removed order 36\n")
fit_results_file.write(str(gc_result_masked_sl)+"\n")
fit_results_file.write(str(sigma_sl)+"\n\n\n")
fit_results_file.close()



multipage("/u/ghezgroup/data/metallicity/nirspec/spectra_fits/pdfs/NGC6791_J19205+3748282_order36_sl<"+str(sys.argv[1])+"_cut_fixed_vrad.pdf")
