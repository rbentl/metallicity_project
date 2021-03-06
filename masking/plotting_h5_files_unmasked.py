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

unmasked_wavelength, unmasked_flux = model()

gc_result = MultiNestResult.from_hdf5("/u/rbentley/metallicity/spectra_fits/masked_fit_results/sensitivity_cut_0_NGC6791_J19205+3748282_order35.h5")

print gc_result

gc_sigmas = gc_result.calculate_sigmas(1)

for a in gc_result.median.keys():
    setattr(model,a,gc_result.median[a])

sl_wavelength, sl_flux = model()

data_unmasked_residual = mt.calc_residuals(sl_flux,starspectrum35.flux.value)

residual_mask_flux = []
residual_mask_wavelength = []

f= plt.figure(figsize=(12,9))
dataax  = f.add_subplot(4,1,1)
modelax  = f.add_subplot(4,1,2,sharex=dataax)
residualax  = f.add_subplot(4,1,3)
tableax = f.add_subplot(4,1,4)


l1 = dataax.plot(starspectrum35.wavelength.value, starspectrum35.flux.value, color='blue')[0]

dataax.set_xticks([])

l2 = modelax.plot(sl_wavelength, sl_flux, color='green')[0]

modelax.set_xticks([])

modelax.set_ylabel("Flux",size=14)

dataax.set_title("NGC6791_J19205+3748282 Order 35 with no mask\nNumber of flagged points:"+str(len(residual_mask_flux)))

l3 = residualax.plot(sl_wavelength, data_unmasked_residual, color='0.5')[0]

l3b = residualax.axhline(0.05, color='black',ls='--')
l3c = residualax.axhline(-0.05, color='black',ls='--')

columns = ["$T_{eff}$","Log g","[M/H]","[alpha/Fe]"]

rows = ["Unmasked","APOGEE"]

tableax.axis("tight")
tableax.axis("off")

fitted_params = [str(round(gc_result.median[a],4))+"$\pm$"+str(round((gc_sigmas[a][1]*0.5-gc_sigmas[a][0]*0.5),4)) for a in ["teff_0","logg_0","mh_0","alpha_0"]]

apogee_params = ["4026.56$\pm$46.36","1.653$\pm$0.042","0.4470$\pm$0.0158","0.0201$\pm$0.0086"]

table = tableax.table(cellText=[fitted_params,apogee_params],rowLabels=rows,colLabels=columns, loc='upper center',fontsize=24)
table.scale(1,1.5)

f.legend([l1, l2, l3, l3b],["Data",'Masked Model', 'Unmasked Model-Masked Model Residuals', '$\pm$5% flux'], loc=(0.4,0.02))

#plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])

plt.show()
f.savefig('unmasked_v2.png')
