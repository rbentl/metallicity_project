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
from specutils import read_fits_file,plotlines,write_spectrum
import numpy as np
import os,scipy
from scipy import signal
from specutils import Spectrum1D,rvmeasure
from starkit.fix_spectrum1d import SKSpectrum1D
import datetime,glob
import gc
from matplotlib.backends.backend_pdf import PdfPages

from astropy.modeling import Model

import sys
import os

def closest (num, arr):
    curr = 0
    for index in range (len (arr)):
        if abs (num - arr[index]) < abs (num - arr[curr]):
            curr = index
           
    return curr

def run_multinest_fit(fit_model):
    
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

    
    # summary statistics like the mean and median can be accessed as dictionaries
    print(gc_result.median)
    print(gc_result.mean), "end"

    # can also compute 1 sigma intervals (or arbitrary)
    gc_result.calculate_sigmas(1)

    return gc_result

def run_multinest_fit_rv_constrained(fit_model, vrad, vrad_err):
    
    teff_prior = priors.UniformPrior(1000,6000)
    logg_prior = priors.UniformPrior(0.1,4.0)
    mh_prior = priors.UniformPrior(-2.0,1.0)
    alpha_prior = priors.UniformPrior(-0.25,0.5)
    vrot_prior = priors.UniformPrior(0,350.0)
    #vrad_prior1 = priors.FixedPrior(vrad)
    vrad_prior1 = priors.UniformPrior(vrad-3.*vrad_err,vrad+3.*vrad_err)
    #R_prior1 = priors.UniformPrior(1500,5000)
    R_prior1 = priors.FixedPrior(24000)
    
    # make a MultiNest fitting object with the model and the prior
    gc_fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior,vrot_prior, vrad_prior1,R_prior1])
    # Run the fit using the MultiNest sampler.
    # Will take about 5-10 minutes to run for high resolution Phoenix grid
    gc_result = gc_fitobj.run()

    
    # summary statistics like the mean and median can be accessed as dictionaries
    print(gc_result.median)
    print(gc_result.mean), "end"

    # can also compute 1 sigma intervals (or arbitrary)
    gc_result.calculate_sigmas(1)

    return gc_result



def calc_residuals(best_fit_flux, obs_flux):
    print len(best_fit_flux), len(obs_flux)
    if len(best_fit_flux) != len(obs_flux):
        best_fit_flux = signal.resample(best_fit_flux, len(obs_flux))

    residuals = obs_flux - best_fit_flux

    return residuals


def residual_masked_data(flux,wavelength,uncert,residuals,limit):

    new_f = np.array([])
    new_w = np.array([])
    new_u = np.array([])
    
    for i in range(len(residuals)):
        if abs(residuals[i]) < limit:
            print flux[i], len(new_f)
            new_f = np.append(new_f, flux[i])
            new_w = np.append(new_w, wavelength[i])
            new_u = np.append(new_u, uncert[i])
        


    print len(flux), len(new_f)
    return new_f, new_w, new_u


def find_slopes(model, param, param_vals):
    w_all = []
    f_all = []
    R_all = []
   
    for i in range(len(param_vals)):
        if param is 'teff':
            model.teff_0 = param_vals[i]
        elif param is 'logg':
            model.logg_0 = param_vals[i]
        elif param is 'mh':
            model.mh_0 = param_vals[i]

        w,f = model()
        R_all += [(np.amax(f) - f)/np.amax(f)]
        w_all += [w]
        f_all += [f]

    #print f
    slopes = []
    for j in range(len(w)):
        dflux = []
        if len(param_vals) > 1:
            for i in range(len(param_vals)):
                dflux += [f_all[i][j] - f_all[0][j]]
            fit_params = np.polyfit(param_vals, dflux, 1)
            slopes += [fit_params[0]]


    return slopes


def slope_masked_data(wavelength, flux, uncert, slopes, limit):
    new_flux = np.array([])
    new_wave = np.array([])
    new_uncert = np.array([])

    
    
    for i in range(len(flux)):
        if abs(slopes[i]) < limit:
            #print flux[i], len(new_flux)
            new_flux = np.append(new_flux, flux[i])
            new_wave = np.append(new_wave, wavelength[i])
            new_uncert = np.append(new_uncert, uncert[i])
        

    return new_flux, new_wave, new_uncert

def r_val(model, param=None, param_val=None):
    
    if param is 'teff':
        model.teff_0 = param_val
    elif param is 'logg':
        model.logg_0 = param_val
    elif param is 'mh':
        model.mh_0 = param_val

    w,f = model()
    R = 100*(np.amax(f) - f)/np.amax(f)
    R0 = 100*(1.-f)/1.

    return R,R0,np.amax(f)


def r_val_polynomial(model):

    w,f = model()

    p = np.polyfit(w,f,3)

    continuum = p[0]*w**3 + p[1]*w**2 + p[2]*w + p[3]

    R = [(continuum[i] - f[i])/continuum[i] for i in range(len(f))]

    return R


def s_lambda(model, param, param_val,increment):
    
    if param is 'teff':
        model.teff_0 = param_val
    elif param is 'logg':
        model.logg_0 = param_val
    elif param is 'mh':
        model.mh_0 = param_val

    R_cen = r_val_polynomial(model)

    if param is 'teff':
        model.teff_0 = param_val+increment
    elif param is 'logg':
        model.logg_0 = param_val+increment
    elif param is 'mh':
        model.mh_0 = param_val+increment

    R_up = r_val_polynomial(model)

    if param is 'teff':
        model.teff_0 = param_val-increment
    elif param is 'logg':
        model.logg_0 = param_val-increment
    elif param is 'mh':
        model.mh_0 = param_val-increment

    R_dw = r_val_polynomial(model)
    
    s_lambda = [100*(R_up[i] - R_dw[i])/R_cen[i] for i in range(len(R_up))]

    return s_lambda
    
def get_snr(starname, order):
    fil = open('/u/ghezgroup/data/metallicity/nirspec/spectra/order_snr.txt')
    table = fil.readlines()

    for i in range(len(table)):
        if starname in table[i] and 'order'+order in table[i]:
            return float(table[i].split()[2])


    print "No SNR found for order "+order+" for star "+starname
    return None


def fit_star_multi_order(starname,g,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                         R_fixed=None,logg_fixed=None,l1norm=False):

    # fit a spectrum of a star with multiple orders that can have different velocities
    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=[2.245, 2.275])
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=[2.181, 2.2103])
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=[2.1168, 2.145])
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
    rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34,nnorm)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(starspectrum35,nnorm)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(starspectrum36,nnorm)



    model = g | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
         norm1 & norm2 & norm3

    w1,f1,w2,f2,w3,f3 = model()


    plt.clf()
    plt.plot(w1,f1)
    plt.plot(starspectrum34.wavelength,starspectrum34.flux)
    plt.plot(w2,f2)
    plt.plot(starspectrum35.wavelength,starspectrum35.flux)

    plt.plot(w3,f3)
    plt.plot(starspectrum36.wavelength,starspectrum36.flux)


    # likelihoods
    if l1norm:
        like1 = L1Likelihood(starspectrum34)
        like2 = L1Likelihood(starspectrum35)
        like3 = L1Likelihood(starspectrum36)

    else:
        like1 = Chi2Likelihood(starspectrum34)
        like2 = Chi2Likelihood(starspectrum35)
        like3 = Chi2Likelihood(starspectrum36)

    fit_model = model | like1 & like2 & like3 | Combiner3()
    print fit_model.__class__
    print fit_model()

    teff_prior = priors.UniformPrior(*teff_range)
    if logg_fixed is not None:
        logg_prior = priors.FixedPrior(logg_fixed)
    else:
        logg_prior = priors.UniformPrior(*logg_range)
    mh_prior = priors.UniformPrior(*mh_range)
    alpha_prior = priors.UniformPrior(*alpha_range)
    vrot_prior = priors.UniformPrior(*vrot_range)
    vrad_prior1 = priors.UniformPrior(*vrad_range)
    vrad_prior2 = priors.UniformPrior(*vrad_range)
    vrad_prior3 = priors.UniformPrior(*vrad_range)

    # R_prior1 = priors.FixedPrior(R)
    # R_prior2 = priors.FixedPrior(R)
    # R_prior3 = priors.FixedPrior(R)
    # R_prior4 = priors.FixedPrior(R)

    if R_fixed is not None:
        R_prior1 = priors.FixedPrior(R_fixed)
        R_prior2 = priors.FixedPrior(R_fixed)
        R_prior3 = priors.FixedPrior(R_fixed)
    else:
        R_prior1 = priors.UniformPrior(*r_range)
        R_prior2 = priors.UniformPrior(*r_range)
        R_prior3 = priors.UniformPrior(*r_range)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, \
             vrad_prior1,vrad_prior2,vrad_prior3,R_prior1,R_prior2,\
             R_prior3])

    fitobj.run(verbose=verbose,importance_nested_sampling=False,n_live_points=400)
    result=fitobj.result

    if l1norm:
        like_str = '_l1norm'
    else:
        like_str = ''
    result.to_hdf(os.path.join(savedir,'unmasked_'+starname+'_order34-36'+like_str+'.h5'))
    print result.calculate_sigmas(1)
    print result.median

    # save the individual model spectra with the max posterior value
    model.teff_0 = result.maximum['teff_0']
    model.logg_0 = result.maximum['logg_0']
    model.mh_0 = result.maximum['mh_0']
    model.alpha_0 = result.maximum['alpha_0']
    model.vrot_1 = result.maximum['vrot_1']
    model.vrad_3 = result.maximum['vrad_3']
    model.vrad_4 = result.maximum['vrad_4']
    model.vrad_5 = result.maximum['vrad_5']
    model.R_7 = result.maximum['R_6']
    model.R_8 = result.maximum['R_7']
    model.R_9 = result.maximum['R_8']

    w1,f1,w2,f2,w3,f3 = model()

    comment1 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_3.value,model.R_7.value)
    comment2 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_4.value,model.R_8.value)
    comment3 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_5.value,model.R_9.value)

    file1 = os.path.join(savedir,starname+'_order34_model.txt')
    file2 = os.path.join(savedir,starname+'_order35_model.txt')
    file3 = os.path.join(savedir,starname+'_order36_model.txt')

    write_spectrum.write_txt(w1,f1,file1,comments=comment1)
    write_spectrum.write_txt(w2,f2,file2,comments=comment2)
    write_spectrum.write_txt(w3,f3,file3,comments=comment3)



class Splitter3(Model):
    # split a single spectrum into 3
    inputs=('w', 'f')
    outputs = ('w', 'f', 'w', 'f','w','f')
    def evaluate(self, w, f):
        return w,f,w,f,w,f
    
class Combiner3(Model):
    # combines the likelihood for four spectra
    inputs=('l1', 'l2','l3')
    outputs = ('ltot',)
    def evaluate(self, l1, l2,l3):
        return l1+l2+l3


def plot_multi_order_fit(starname,g=None,savefile=None,specdir='/group/data/nirspec/spectra/',
                         savedir = '../nirspec_fits/',snr=30.0,nnorm=2,save_model=False,plot_maximum=False):
    # plot the results of a multiple order fit on observed spectrum.

    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')

    if savefile is None:
        savefile = os.path.join(savedir,'unmasked_'+starname+'_order34-36'+like_str+'.h5')
    # restore MultiNest savefile
    result = MultiNestResult.from_hdf5(savefile)


    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=[2.245, 2.275])
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=[2.181, 2.2103])
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=[2.1168, 2.145])
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    if g is not None:
        interp1 = Interpolate(starspectrum34)
        convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
        rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm1 = Normalize(starspectrum34,nnorm)

        interp2 = Interpolate(starspectrum35)
        convolve2 = InstrumentConvolveGrating.from_grid(g,R=24000)
        #rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm2 = Normalize(starspectrum35,nnorm)

        interp3 = Interpolate(starspectrum36)
        convolve3 = InstrumentConvolveGrating.from_grid(g,R=24000)
        #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm3 = Normalize(starspectrum36,nnorm)


        model = g | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
             convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
             norm1 & norm2 & norm3

        if plot_maximum:
            model.teff_0 = result.maximum['teff_0']
            model.logg_0 = result.maximum['logg_0']
            model.mh_0 = result.maximum['mh_0']
            model.alpha_0 = result.maximum['alpha_0']
            model.vrot_1 = result.maximum['vrot_1']
            model.vrad_3 = result.maximum['vrad_3']
            model.vrad_4 = result.maximum['vrad_4']
            model.vrad_5 = result.maximum['vrad_5']
            model.R_6 = result.maximum['R_6']
            model.R_7 = result.maximum['R_7']
            model.R_8 = result.maximum['R_8']
            #model.R_9 = result.median['R_9']

        else:
            model.teff_0 = result.median['teff_0']
            model.logg_0 = result.median['logg_0']
            model.mh_0 = result.median['mh_0']
            model.alpha_0 = result.median['alpha_0']
            model.vrot_1 = result.median['vrot_1']
            model.vrad_3 = result.median['vrad_3']
            model.vrad_4 = result.median['vrad_4']
            model.vrad_5 = result.median['vrad_5']
            model.R_6 = result.median['R_6']
            model.R_7 = result.median['R_7']
            model.R_8 = result.median['R_8']
            #model.R_9 = result.median['R_9']

        w1,f1,w2,f2,w3,f3 = model()

    else:

        file1 = os.path.join(savedir,starname+'_order34_model.txt')
        file2 = os.path.join(savedir,starname+'_order35_model.txt')
        file3 = os.path.join(savedir,starname+'_order36_model.txt')

        w1,f1 = np.loadtxt(file1,usecols=(0,1),unpack=True)
        w2,f2 = np.loadtxt(file2,usecols=(0,1),unpack=True)
        w3,f3 = np.loadtxt(file3,usecols=(0,1),unpack=True)


    # teff_0      3363.211996
    # logg_0         1.691725
    # mh_0           0.936003
    # alpha_0       -0.027917
    # vrot_1         1.378488
    # vrad_3      -538.550269
    # vrad_4      -239.851862
    # vrad_5      -541.044943
    # vrad_6      -540.432821
    # R_7        20000.000000
    # R_8        20000.000000
    # R_9        20000.000000
    # R_10       20000.000000

    plt.clf()
    observed_wave = (starspectrum36.wavelength,
                     starspectrum35.wavelength,starspectrum34.wavelength)
    observed_flux = (starspectrum36.flux,
                     starspectrum35.flux,starspectrum34.flux)
    model_wave = (w3,w2,w1)
    model_flux = (f3,f2,f1)
    max_result = result.maximum
    vels = (max_result['vrad_5'],max_result['vrad_4'],max_result['vrad_3'])

    print 'maximum likelihood:'
    print max_result

    print 'median:'
    print result.median
    
    print '1 sigma:'
    print result.calculate_sigmas(1)
    
    for i in xrange(len(observed_wave)):
        plt.subplot(4,1,i+1)
        velfac = 1.0/(vels[i]/3e5+1.0)
        xwave = observed_wave[i]*velfac
        plt.plot(xwave,observed_flux[i])
        plt.plot(model_wave[i]*velfac,model_flux[i])
        plt.ylim(0.2,1.2)
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Flux')
        plt.xlim(np.min(xwave.value),np.max(xwave.value))
        plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,molecules=False,size=12)

    plt.tight_layout()
    plt.show()
    if save_model:
        comment1 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
                   (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
                   model.vrot_1.value,model.vrad_3.value,model.R_6.value)
        comment2 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
                   (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
                   model.vrot_1.value,model.vrad_4.value,model.R_7.value)
        comment3 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
                   (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
                   model.vrot_1.value,model.vrad_5.value,model.R_8.value)

        file1 = os.path.join(savedir,starname+'_order34_model.txt')
        file2 = os.path.join(savedir,starname+'_order35_model.txt')
        file3 = os.path.join(savedir,starname+'_order36_model.txt')

        write_spectrum.write_txt(w1,f1,file1,comments=comment1)
        write_spectrum.write_txt(w2,f2,file2,comments=comment2)
        write_spectrum.write_txt(w3,f3,file3,comments=comment3)




