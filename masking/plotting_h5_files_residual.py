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



interp1 = Interpolate(starspectrum35)
convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
norm1 = Normalize(starspectrum35,2)

# concatenate the spectral grid (which will have the stellar parameters) with other 
# model components that you want to fit
model = g | rot1 |DopplerShift(vrad=0.0)| convolve1 | interp1 | norm1


apogee_vals = {"teff_0":4026.56,
                        "logg_0":1.653,
                        "mh_0":0.447044,
                        "alpha_0":0.020058,
                        "vrot_1":0.928,
                        "vrad_2":-48.4008,
                        "R_3":24000.}

for a in apogee_vals.keys():
    setattr(model,a,apogee_vals[a])


cutoff = 0.05
cutoff_str = str(cutoff)

unmasked_wavelength, unmasked_flux = model()

gc_result = MultiNestResult.from_hdf5("/u/rbentley/metallicity/spectra_fits/masked_fit_results/residual_cut_005_NGC6791_J19205+3748282_order35.h5")

print gc_result

gc_sigmas = gc_result.calculate_sigmas(1)

for a in gc_result.median.keys():
    setattr(model,a,gc_result.median[a])

sl_wavelength, sl_flux = model()

data_unmasked_residual = mt.calc_residuals(sl_flux,starspectrum35.flux.value)

masked_unmasked_residual = mt.calc_residuals(sl_flux,unmasked_flux)

residual_mask_flux = []
residual_mask_wavelength = []

for i in range(len(data_unmasked_residual)):
    if abs(data_unmasked_residual[i]) > cutoff:
        residual_mask_flux += [starspectrum35.flux.value[i]]
        residual_mask_wavelength += [starspectrum35.wavelength.value[i]]
        

#f, (dataax, modelax, residualax) = plt.subplots(4, sharex=True)

f= plt.figure(figsize=(12,9))
dataax  = f.add_subplot(3,1,1)
modelax  = f.add_subplot(3,1,2,sharex=dataax)
tableax = f.add_subplot(3,1,3)

fig2 = plt.figure(figsize=(12,9))
modelax2  = fig2.add_subplot(4,1,1)
dataresidualax  = fig2.add_subplot(4,1,2,sharex=dataax)
modelresidualax  = fig2.add_subplot(4,1,3)
tableax2 = fig2.add_subplot(4,1,4)


l1 = dataax.plot(starspectrum35.wavelength.value/(gc_result.median['vrad_2']/3e5+1.0), starspectrum35.flux.value, color='blue')[0]
l1b = dataax.plot(residual_mask_wavelength/(gc_result.median['vrad_2']/3e5+1.0), residual_mask_flux, "ro", markersize=2.5)[0]

dataax.set_xticks([])

l2 = modelax.plot(sl_wavelength/(gc_result.median['vrad_2']/3e5+1.0), sl_flux, color='green')[0]

modelax.set_xticks([])

modelax.set_ylabel("Flux",size=14)

dataax.set_title("NGC6791_J19205+3748282 Order 35 with residual < "+cutoff_str+" masked fit\nNumber of flagged points:"+str(len(residual_mask_flux)))

columns = ["$T_{eff}$","Log g","[M/H]","[alpha/Fe]"]

rows = ["Masked","Unmasked","APOGEE"]

tableax.axis("tight")
tableax.axis("off")

fitted_params = [str(round(gc_result.median[a],4))+"$\pm$"+str(round((gc_sigmas[a][1]*0.5-gc_sigmas[a][0]*0.5),4)) for a in ["teff_0","logg_0","mh_0","alpha_0"]]

apogee_params = ["4026.56$\pm$46.36","1.653$\pm$0.042","0.4470$\pm$0.0158","0.0201$\pm$0.0086"]

unmasked_params = ["3891.37$\pm$8.85","0.104$\pm$0.005","0.003$\pm$0.043","0.306$\pm$0.020"]


table = tableax.table(cellText=[fitted_params,unmasked_params,apogee_params],rowLabels=rows,colLabels=columns, loc='upper center',fontsize=24)
table.scale(1,1.5)

f.legend([l1, l1b, l2],["Data","Masked Data Points",'Masked Model'], loc=(0.4,0.02))





l4 = modelax2.plot(sl_wavelength/(gc_result.median['vrad_2']/3e5+1.0), sl_flux, color='green')[0]

l4b = modelax2.plot(unmasked_wavelength/(gc_result.median['vrad_2']/3e5+1.0), unmasked_flux, color='red', alpha=0.3)[0]

modelax2.set_xticks([])

l5 = dataresidualax.plot(sl_wavelength/(gc_result.median['vrad_2']/3e5+1.0), data_unmasked_residual, color='0.5')[0]

l5b = dataresidualax.axhline(0.05, color='black',ls='--')
l5c = dataresidualax.axhline(-0.05, color='black',ls='--')

dataresidualax.set_xticks([])

dataresidualax.set_ylabel("Flux",size=14)

l6 = modelresidualax.plot(sl_wavelength/(gc_result.median['vrad_2']/3e5+1.0), masked_unmasked_residual, color='blue')[0]

l6b = modelresidualax.axhline(0.05, color='black',ls='--')
l6c = modelresidualax.axhline(-0.05, color='black',ls='--')

#residualax.set_xlabel("Wavelength (Angstrom)",size=14)

modelax2.set_title("NGC6791_J19205+3748282 Order 35 with residual < "+cutoff_str+" masked fit\nNumber of flagged points:"+str(len(residual_mask_flux)))

tableax2.axis("tight")
tableax2.axis("off")


table2 = tableax2.table(cellText=[fitted_params,unmasked_params,apogee_params],rowLabels=rows,colLabels=columns, loc='upper center',fontsize=24)
table2.scale(1,1.5)


fig2.legend([l4, l4b, l5, l6, l6b],['Masked Model', 'Unmasked Model','Masked Model-Data Residuals', 'Unmasked Model-Masked Model Residuals', '$\pm$5% flux'], loc=(0.4,0.02))

#plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])

plt.show()
f.savefig('residual_masked_'+cutoff_str+'_v2a.png')
fig2.savefig('residual_masked_'+cutoff_str+'_v2b.png')
