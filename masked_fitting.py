import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
from specutils import read_fits_file,plotlines
import numpy as np
import os,scipy
from scipy import signal
from specutils import Spectrum1D,rvmeasure
import datetime,glob
import gc
from matplotlib.backends.backend_pdf import PdfPages

import sys
import os


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


# try:
#     import MySQLdb as mdb
# except:
#     import pymysql as mdb

specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
savedir = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masks/'

starname = 'TYC 3544'
order = str(33)
radv = 0.0
teff = 5556.
logg = 1.
mh = -0.02
file2 = glob.glob(specdir+starname+'_order'+order+'*.dat')
print file2
snr = float(53.663)
    
starspectrumall = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='micron')
    
waverange = [np.amin(starspectrumall.wavelength.value[:970]), np.amax(starspectrumall.wavelength.value[:970])]
starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',
                                                 wave_range=waverange)
starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit


savefile = os.path.join(savedir,starname+'order'+order+'_phoenixr40000.h5')

print 'here 1 in file'

# load the BOSZ grid. Do this only ONCE!! It takes a long time and lots of memory
g = load_grid('/u/rbentley/metallicity/spectra_fits/phoenix_t2500_6000_w20000_24000_R40000.h5')#'/u/rbentley/metallicity/spectra_fits/test_bosz_t2500_6000_w20000_24000_R40000.h5')
print 'grid loaded'






interp1 = Interpolate(starspectrum35)
print 'interpolated'
convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
print 'convolved'
rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
print 'rot broadend'
norm1 = Normalize(starspectrum35,2)
print 'normalized'
# concatenate the spectral grid (which will have the stellar parameters) with other
# model components that you want to fit
model = g | rot1 |DopplerShift(vrad=radv)| convolve1 | interp1 | norm1
print 'model concaten'



w,f = model()

#plt.plot(w,f)
    
model.teff_0 = teff
model.logg_0 = logg
model.mh_0 = mh
w,f = model()
wref, fref = model()

resampled_apogeew = (signal.resample(wref, len(starspectrum35.wavelength.value)))/(radv/3e5+1.0)
resampled_apogeef = signal.resample(fref, len(starspectrum35.flux.value))

apogee_res = starspectrum35.flux.value - resampled_apogeef
apogee_mask = []
apogee_mask_w = []
apogee_ind = []
for i in range(len(apogee_res)):
    if abs(apogee_res[i])>0.05:
        apogee_mask += [apogee_res[i]]
        apogee_mask_w += [resampled_apogeew[i]]
        apogee_ind += [i]




apogee_ind_s = sorted(apogee_ind, reverse=True)
print 'trimming'
starspectrum_fmasked = np.delete(starspectrum35.flux,apogee_ind_s)
starspectrum_wmasked = np.delete(starspectrum35.wavelength,apogee_ind_s)
starspectrum_umasked = np.delete(starspectrum35.uncertainty,apogee_ind_s)
print len(starspectrum_fmasked), len(apogee_ind)

spectrum_masked = Spectrum1D.from_array(starspectrum_wmasked, starspectrum_fmasked, dispersion_unit=starspectrum35.wavelength.unit, uncertainty=starspectrum_umasked) 
                                        #flux_unit=starspectrum35.flux.unit, wavelength_unit=starspectrum35.wavelength.unit)

print starspectrum_fmasked
interp1 = Interpolate(spectrum_masked)
print 'interpolated 2'
convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
print 'convolved 2'
rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
print 'rot broadend 2'
norm1 = Normalize(spectrum_masked,2)
print 'normalized 2'
# concatenate the spectral grid (which will have the stellar parameters) with other
# model components that you want to fit
model = g | rot1 |DopplerShift(vrad=radv)| convolve1 | interp1 | norm1
print 'model concaten 2'

masked_w,masked_f = model()
    
#for i in sorted(apogee_ind, reverse=True):
#    print i
#    np.delete(starspectrum35.wavelength, i)
#    np.delete(starspectrum35.flux, i)
#del starspectrum35.wavelength[i]
# add likelihood parts
like1 = Chi2Likelihood(spectrum_masked)
#like1_l1 = SpectralL1Likelihood(spectrum)



fit_model = model | like1
print 'here 2', spectrum_masked.wavelength.unit
# look at parameters in the model
model
    
'''
plt.plot(w,f)
plt.plot(starspectrum35.wavelength,starspectrum35.flux)
'''



#plt.savefig('testfig1.pdf')
print(fit_model())





teff_prior = priors.UniformPrior(1000,6000)
logg_prior = priors.UniformPrior(0.1,4.0)
mh_prior = priors.UniformPrior(-2.0,1.0)
alpha_prior = priors.UniformPrior(-0.25,0.5)
vrot_prior = priors.UniformPrior(0,350.0)
vrad_prior1 = priors.UniformPrior(-1000,1000)
#R_prior1 = priors.UniformPrior(1500,5000)
R_prior1 = priors.FixedPrior(24000)

# make a MultiNest fitting object with the model and the prior
gc_fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior,vrot_prior, vrad_prior1,R_prior1])





# Run the fit using the MultiNest sampler.
# Will take about 5-10 minutes to run for high resolution Phoenix grid
gc_result = gc_fitobj.run()

# after fitting, save the results to an HDF5 file
gc_result.to_hdf(savefile)
print "here"
# load from the HDF5 file in case the fit was not run
gc_result = MultiNestResult.from_hdf5(savefile)




# summary statistics like the mean and median can be accessed as dictionaries
print(gc_result.median)
print(gc_result.mean), "end"

# can also compute 1 sigma intervals (or arbitrary)
gc_result.calculate_sigmas(1)

# can also make corner plots
# be sure to specify parameters if one of them is fixed - if not, corner will crash
gc_result.plot_triangle(parameters=['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2'])

# set model to median values
for a in gc_result.median.keys():
    setattr(model,a,gc_result.median[a])



w,f = model()

#print "chi squared val ", like1

residual_f = []
residual_w = []


#resampled_w = (signal.resample(starspectrum35.wavelength.value, len(w)))/(gc_result.median['vrad_2']/3e5+1.0)
#resampled_f = signal.resample(starspectrum35.flux.value, len(f))

resampled_mw = (signal.resample(w, len(starspectrum35.wavelength.value)))/(gc_result.median['vrad_2']/3e5+1.0)
resampled_mf = signal.resample(f, len(starspectrum35.flux.value))
        
residual_f = starspectrum35.flux.value - resampled_mf
print len(resampled_mf)





plt.figure(figsize=(15,7))
plt.plot(starspectrum35.wavelength.value/(gc_result.median['vrad_2']/3e5+1.0),starspectrum35.flux, label='Data')
plt.plot(w/(gc_result.median['vrad_2']/3e5+1.0),f,label='Fitted Model')
plt.plot(resampled_mw,residual_f,label='Data-Fitted Model Residuals')
plt.axhline(y=0.03, color='r', linestyle='-')
plt.axhline(y=-0.03, color='r', linestyle='-')
plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux')
plt.legend(loc='center right')

model_residuals = f-fref

plt.figure(figsize=(15,7))
plt.plot(starspectrum35.wavelength.value/(gc_result.median['vrad_2']/3e5+1.0),starspectrum35.flux, label='Data')
plt.plot(apogee_mask_w,apogee_mask, label='APOGEE Mask', marker='o')
plt.plot(resampled_apogeew,apogee_res, label='APOGEE Res')

#plt.plot(w/(gc_result.median['vrad_2']/3e5+1.0),f,label='Fitted Model')
#plt.plot(wref/(float(radv)/3e5+1.0),fref,label='Reference Model')
#plt.plot(wref/(float(radv)/3e5+1.0),model_residuals,label='Fitted Model-Ref Model Residuals')
plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,size=6,highlight=['Sc'])
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux')
plt.legend(loc='center right')
    
#multipage(starname+'_order'+order+'_fixed_logg_mh.pdf')

absresidual = abs(residual_f)

maxres = np.amax(absresidual)

meanres = np.mean(absresidual)

redchi2array = absresidual/starspectrum35.uncertainty.value
print absresidual[0:70]
print starspectrum35.uncertainty.value[0:70]
redchi2array = redchi2array**2
redchi2 = sum(redchi2array)/len(redchi2array)
print redchi2array[0:70]
print len(redchi2array)
print maxres, meanres,redchi2, "max residual and mean residual"

plt.show()
