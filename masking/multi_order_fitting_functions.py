#from starkit.gridkit.io.phoenix.base import PhoenixSpectralGridIO, ParameterSet
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood
from starkit.fitkit.likelihoods import SpectralL1Likelihood as L1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest,MultiNestResult
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from starkit import assemble_model, operations
from starkit.fitkit import priors
from specutils import read_fits_file,plotlines,write_spectrum
import numpy as np
import os,scipy
from specutils import Spectrum1D,rvmeasure
import pylab as plt
import glob
#import seaborn as sns
from astropy.modeling import models,fitting
import astropy.units as u
from astropy.modeling import models,fitting
from astropy.modeling import Model
import matplotlib
import pandas as pd
import collections
font = {        'size'   : 16}
matplotlib.rc('font', **font)

import model_tester_updated as mtu

#sns.set_context('paper',font_scale=2.0, rc={"lines.linewidth": 1.75})
#sns.set_style("white")
#sns.set_style('ticks')


apogee_vals = {"teff_0":4026.56,
                        "logg_0":1.653,
                        "mh_0":0.447044,
                        "alpha_0":0.020058,
                        "vrot_1":0.928#,
                        #"vrad_2":-48.4008,
                        #"R_3":24000.
}




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

class Splitter4(Model):
    # split a single spectrum into 4
    inputs=('w', 'f')
    outputs = ('w', 'f', 'w', 'f','w','f','w','f')
    def evaluate(self, w, f):
        return w,f,w,f,w,f,w,f

class Combiner4(Model):
    # combines the likelihood for four spectra
    inputs=('l1', 'l2','l3','l4')
    outputs = ('ltot',)
    def evaluate(self, l1, l2,l3,l4):
        return l1+l2+l3+l4
    

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



def plot_multi_order(starname,g,specdir='/group/data/nirspec/spectra/',
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





def fit_star_three_orders_masked(starname,g,sl_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                         R_fixed=None,logg_fixed=None,l1norm=False):


    # fit a spectrum of a star with multiple orders that can have different velocities
    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')
    
    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='micron')
    
    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]    
    
    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    #g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

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
    
    sl_mh1,sl_mh2,sl_mh3 = s_lambda_three_order(model,'mh',model.mh_0.value,0.1)

    w1,f1,w2,f2,w3,f3 = model()


    #plt.clf()
    #plt.plot(w1,f1)
    #plt.plot(starspectrum34.wavelength,starspectrum34.flux)
    #plt.plot(w2,f2)
    #plt.plot(starspectrum35.wavelength,starspectrum35.flux)

    #plt.plot(w3,f3)
    #plt.plot(starspectrum36.wavelength,starspectrum36.flux)


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


    #starspectrumall_flux = np.concatenate((starspectrum34.flux.value,starspectrum35.flux.value,starspectrum36.flux.value))
    #starspectrumall_wavelength = np.concatenate((starspectrum34.wavelength.value,starspectrum35.wavelength.value,starspectrum36.wavelength.value))
    #starspectrumall_uncert = np.concatenate((starspectrum34.uncertainty.value,starspectrum35.uncertainty.value,starspectrum36.uncertainty.value))

    sl_mask_indices1 = []
    sl_mask_indices2 = []
    sl_mask_indices3 = []
    
    for i in range(len(sl_mh1)):
        if abs(sl_mh1[i])<float(sl_cut):
            sl_mask_indices1 += [i]

    for i in range(len(sl_mh2)):
        if abs(sl_mh2[i])<float(sl_cut):
            sl_mask_indices2 += [i]

    for i in range(len(sl_mh3)):
        if abs(sl_mh3[i])<float(sl_cut):
            sl_mask_indices3 += [i]
            

    masked_data_sl_f1 = np.delete(starspectrum34.flux.value,sl_mask_indices1)
    masked_data_sl_w1 = np.delete(starspectrum34.wavelength.value,sl_mask_indices1)
    masked_data_sl_u1 = np.delete(starspectrum34.uncertainty.value,sl_mask_indices1)

    masked_data_sl_f2 = np.delete(starspectrum35.flux.value,sl_mask_indices2)
    masked_data_sl_w2 = np.delete(starspectrum35.wavelength.value,sl_mask_indices2)
    masked_data_sl_u2 = np.delete(starspectrum35.uncertainty.value,sl_mask_indices2)

    masked_data_sl_f3 = np.delete(starspectrum36.flux.value,sl_mask_indices3)
    masked_data_sl_w3 = np.delete(starspectrum36.wavelength.value,sl_mask_indices3)
    masked_data_sl_u3 = np.delete(starspectrum36.uncertainty.value,sl_mask_indices3)
    print "Number of included data points for one order:",len(masked_data_sl_w1)
    masked_data_sl1 = Spectrum1D.from_array(dispersion=masked_data_sl_w1, flux=masked_data_sl_f1, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u1)
    masked_data_sl2 = Spectrum1D.from_array(dispersion=masked_data_sl_w2, flux=masked_data_sl_f2, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u2)
    masked_data_sl3 = Spectrum1D.from_array(dispersion=masked_data_sl_w3, flux=masked_data_sl_f3, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u3)
    

    sl_interp1 = Interpolate(masked_data_sl1)
    sl_convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
    sl_rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm1 = Normalize(masked_data_sl1,nnorm)

    sl_interp2 = Interpolate(masked_data_sl2)
    sl_convolve2 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm2 = Normalize(masked_data_sl2,nnorm)

    sl_interp3 = Interpolate(masked_data_sl3)
    sl_convolve3 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm3 = Normalize(masked_data_sl3,nnorm)


    model = g | sl_rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         sl_convolve1 & sl_convolve2 & sl_convolve3 | sl_interp1 & sl_interp2 & sl_interp3 | \
         sl_norm1 & sl_norm2 & sl_norm3


    w1,f1,w2,f2,w3,f3 = model()

    if l1norm:
        sl_like1 = L1Likelihood(masked_data_sl1)
        sl_like2 = L1Likelihood(masked_data_sl2)
        sl_like3 = L1Likelihood(masked_data_sl3)

    else:
        sl_like1 = Chi2Likelihood(masked_data_sl1)
        sl_like2 = Chi2Likelihood(masked_data_sl2)
        sl_like3 = Chi2Likelihood(masked_data_sl3)

    fit_model = model | sl_like1 & sl_like2 & sl_like3 | Combiner3()
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
    result.to_hdf(os.path.join(savedir,'masked_sl_cutoff_'+str(sl_cut)+'_'+starname+'_order34-36'+like_str+'.h5'))
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
                model.vrot_1.value,model.vrad_3.value,model.R_6.value)
    comment2 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
                model.vrot_1.value,model.vrad_4.value,model.R_7.value)
    comment3 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
                model.vrot_1.value,model.vrad_5.value,model.R_8.value)

    file1 = os.path.join(savedir,'textoutput/order34/'+starname+'_order34_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file2 = os.path.join(savedir,'textoutput/order35/'+starname+'_order35_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file3 = os.path.join(savedir,'textoutput/order36/'+starname+'_order36_model_sl_cutoff_'+str(sl_cut)+'.txt')

    write_spectrum.write_txt(w1,f1,file1,comments=comment1)
    write_spectrum.write_txt(w2,f2,file2,comments=comment2)
    write_spectrum.write_txt(w3,f3,file3,comments=comment3)



def fit_star_four_orders_masked(starname,g,sl_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                         R_fixed=None,logg_fixed=None,l1norm=False):


    # fit a spectrum of a star with multiple orders that can have different velocities
    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')
    file4 = glob.glob(specdir+starname+'_order37*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='micron')
    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='micron')
    
    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]    
    waverange37 = [np.amin(starspectrum37.wavelength.value[:970]), np.amax(starspectrum37.wavelength.value[:970])]
    
    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
    
    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)
    starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum37.flux.unit

    '''
    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=[2.245, 2.275])
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=[2.181, 2.2103])
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=[2.1168, 2.145])
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
    
    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='Angstrom',wave_range=[2.059, 2.087])
    starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum37.flux.unit
    '''
        

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

    interp4 = Interpolate(starspectrum37)
    convolve4 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot4 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm4 = Normalize(starspectrum37,nnorm)


    model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
         norm1 & norm2 & norm3 & norm4

    w1,f1,w2,f2,w3,f3,w4,f4 = model()
    print model

    sl_mh1,sl_mh2,sl_mh3,sl_mh4 = s_lambda_four_order(model,'mh',model.mh_0.value,0.1)

    # likelihoods
    if l1norm:
        like1 = L1Likelihood(starspectrum34)
        like2 = L1Likelihood(starspectrum35)
        like3 = L1Likelihood(starspectrum36)
        like4 = L1Likelihood(starspectrum37)

    else:
        like1 = Chi2Likelihood(starspectrum34)
        like2 = Chi2Likelihood(starspectrum35)
        like3 = Chi2Likelihood(starspectrum36)
        like4 = Chi2Likelihood(starspectrum37)

    fit_model = model | like1 & like2 & like3 & like4 | Combiner4()
    print fit_model.__class__
    print fit_model()

    #starspectrumall_flux = np.concatenate((starspectrum34.flux.value,starspectrum35.flux.value,starspectrum36.flux.value))
    #starspectrumall_wavelength = np.concatenate((starspectrum34.wavelength.value,starspectrum35.wavelength.value,starspectrum36.wavelength.value))
    #starspectrumall_uncert = np.concatenate((starspectrum34.uncertainty.value,starspectrum35.uncertainty.value,starspectrum36.uncertainty.value))

    sl_mask_indices1 = []
    sl_mask_indices2 = []
    sl_mask_indices3 = []
    sl_mask_indices4 = []

    for i in range(len(sl_mh1)):
        if abs(sl_mh1[i])<float(sl_cut):
            sl_mask_indices1 += [i]

    for i in range(len(sl_mh2)):
        if abs(sl_mh2[i])<float(sl_cut):
            sl_mask_indices2 += [i]

    for i in range(len(sl_mh3)):
        if abs(sl_mh3[i])<float(sl_cut):
            print abs(sl_mh3[i])
            sl_mask_indices3 += [i]


    for i in range(len(sl_mh4)):
        if abs(sl_mh4[i])<float(sl_cut):
            sl_mask_indices4 += [i]
 
    masked_data_sl_f1 = np.delete(starspectrum34.flux.value,sl_mask_indices1)
    masked_data_sl_w1 = np.delete(starspectrum34.wavelength.value,sl_mask_indices1)
    masked_data_sl_u1 = np.delete(starspectrum34.uncertainty.value,sl_mask_indices1)

    masked_data_sl_f2 = np.delete(starspectrum35.flux.value,sl_mask_indices2)
    masked_data_sl_w2 = np.delete(starspectrum35.wavelength.value,sl_mask_indices2)
    masked_data_sl_u2 = np.delete(starspectrum35.uncertainty.value,sl_mask_indices2)

    masked_data_sl_f3 = np.delete(starspectrum36.flux.value,sl_mask_indices3)
    masked_data_sl_w3 = np.delete(starspectrum36.wavelength.value,sl_mask_indices3)
    masked_data_sl_u3 = np.delete(starspectrum36.uncertainty.value,sl_mask_indices3)

    masked_data_sl_f4 = np.delete(starspectrum37.flux.value,sl_mask_indices4)
    masked_data_sl_w4 = np.delete(starspectrum37.wavelength.value,sl_mask_indices4)
    masked_data_sl_u4 = np.delete(starspectrum37.uncertainty.value,sl_mask_indices4)
    
    print "Number of included data points for one order:",len(masked_data_sl_w1)
    masked_data_sl1 = Spectrum1D.from_array(dispersion=masked_data_sl_w1, flux=masked_data_sl_f1, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u1)
    masked_data_sl2 = Spectrum1D.from_array(dispersion=masked_data_sl_w2, flux=masked_data_sl_f2, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u2)
    masked_data_sl3 = Spectrum1D.from_array(dispersion=masked_data_sl_w3, flux=masked_data_sl_f3, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u3)
    masked_data_sl4 = Spectrum1D.from_array(dispersion=masked_data_sl_w4, flux=masked_data_sl_f4, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u4)
    
    print sl_mask_indices4
    
    sl_interp1 = Interpolate(masked_data_sl1)
    sl_convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
    sl_rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm1 = Normalize(masked_data_sl1,nnorm)

    sl_interp2 = Interpolate(masked_data_sl2)
    sl_convolve2 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm2 = Normalize(masked_data_sl2,nnorm)

    sl_interp3 = Interpolate(masked_data_sl3)
    sl_convolve3 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm3 = Normalize(masked_data_sl3,nnorm)

    
    sl_interp4 = Interpolate(masked_data_sl4)
    sl_convolve4 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    sl_norm4 = Normalize(masked_data_sl4,nnorm)


    model = g | sl_rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         sl_convolve1 & sl_convolve2 & sl_convolve3 & sl_convolve4 | sl_interp1 & sl_interp2 & sl_interp3 & sl_interp4 | \
         sl_norm1 & sl_norm2 & sl_norm3 & sl_norm4

    sl_mh1,sl_mh2,sl_mh3,sl_mh4 = s_lambda_four_order(model,'mh',model.mh_0.value,0.1)
    w1,f1,w2,f2,w3,f3,w4,f4 = model()

    if l1norm:
        sl_like1 = L1Likelihood(masked_data_sl1)
        sl_like2 = L1Likelihood(masked_data_sl2)
        sl_like3 = L1Likelihood(masked_data_sl3)
        sl_like4 = L1Likelihood(masked_data_sl4)

    else:
        sl_like1 = Chi2Likelihood(masked_data_sl1)
        sl_like2 = Chi2Likelihood(masked_data_sl2)
        sl_like3 = Chi2Likelihood(masked_data_sl3)
        sl_like4 = Chi2Likelihood(masked_data_sl4)

    fit_model = model | sl_like1 & sl_like2 & sl_like3 & sl_like4 | Combiner4()
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
    vrad_prior4 = priors.UniformPrior(*vrad_range)

    # R_prior1 = priors.FixedPrior(R)
    # R_prior2 = priors.FixedPrior(R)
    # R_prior3 = priors.FixedPrior(R)
    # R_prior4 = priors.FixedPrior(R)

    if R_fixed is not None:
        R_prior1 = priors.FixedPrior(R_fixed)
        R_prior2 = priors.FixedPrior(R_fixed)
        R_prior3 = priors.FixedPrior(R_fixed)
        R_prior4 = priors.FixedPrior(R_fixed)
    else:
        R_prior1 = priors.UniformPrior(*r_range)
        R_prior2 = priors.UniformPrior(*r_range)
        R_prior3 = priors.UniformPrior(*r_range)
        R_prior4 = priors.UniformPrior(*r_range)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, \
             vrad_prior1,vrad_prior2,vrad_prior3,vrad_prior4,R_prior1,R_prior2,\
             R_prior3,R_prior4])

    fitobj.run(verbose=verbose,importance_nested_sampling=False,n_live_points=400)
    result=fitobj.result

    if l1norm:
        like_str = '_l1norm'
    else:
        like_str = ''
    result.to_hdf(os.path.join(savedir,'masked_sl_cutoff_'+str(sl_cut)+'_'+starname+'_order34-37'+like_str+'.h5'))
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
    model.vrad_6 = result.maximum['vrad_6']
    model.R_7 = result.maximum['R_7']
    model.R_8 = result.maximum['R_8']
    model.R_9 = result.maximum['R_9']
    model.R_10 = result.maximum['R_10']


    w1,f1,w2,f2,w3,f3,w4,f4 = model()

    comment1 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_3.value,model.R_7.value)
    comment2 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_4.value,model.R_8.value)
    comment3 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_5.value,model.R_9.value)
    comment4 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_6.value,model.R_10.value)

    file1 = os.path.join(savedir,'textoutput/order34/'+starname+'_order34_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file2 = os.path.join(savedir,'textoutput/order35/'+starname+'_order35_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file3 = os.path.join(savedir,'textoutput/order36/'+starname+'_order36_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file4 = os.path.join(savedir,'textoutput/order37/'+starname+'_order37_model_sl_cutoff_'+str(sl_cut)+'.txt')

    write_spectrum.write_txt(w1,f1,file1,comments=comment1)
    write_spectrum.write_txt(w2,f2,file2,comments=comment2)
    write_spectrum.write_txt(w3,f3,file3,comments=comment3)
    write_spectrum.write_txt(w4,f4,file4,comments=comment4)




def r_val_polynomial_three_order(model):

    w1,f1,w2,f2,w3,f3 = model()

    p1 = np.polyfit(w1,f1,3)

    continuum1 = p1[0]*w1**3 + p1[1]*w1**2 + p1[2]*w1 + p1[3]

    R1 = [(continuum1[i] - f1[i])/continuum1[i] for i in range(len(f1))]



    p2 = np.polyfit(w2,f2,3)
        
    continuum2 = p2[0]*w2**3 + p2[1]*w2**2 + p2[2]*w2 + p2[3]

    R2 = [(continuum2[i] - f2[i])/continuum2[i] for i in range(len(f2))]


    p3 = np.polyfit(w3,f3,3)

    continuum3 = p3[0]*w3**3 + p3[1]*w3**2 + p3[2]*w3 + p3[3]

    R3 = [(continuum3[i] - f3[i])/continuum3[i] for i in range(len(f3))]

    
    return R1, R2, R3

def r_val_polynomial_four_order(model):

    w1,f1,w2,f2,w3,f3,w4,f4 = model()


    
    p1 = np.polyfit(w1,f1,3)

    continuum1 = p1[0]*w1**3 + p1[1]*w1**2 + p1[2]*w1 + p1[3]

    R1 = [(continuum1[i] - f1[i])/continuum1[i] for i in range(len(f1))]



    p2 = np.polyfit(w2,f2,3)
        
    continuum2 = p2[0]*w2**3 + p2[1]*w2**2 + p2[2]*w2 + p2[3]

    R2 = [(continuum2[i] - f2[i])/continuum2[i] for i in range(len(f2))]


    p3 = np.polyfit(w3,f3,3)

    continuum3 = p3[0]*w3**3 + p3[1]*w3**2 + p3[2]*w3 + p3[3]

    R3 = [(continuum3[i] - f3[i])/continuum3[i] for i in range(len(f3))]

    

    p4 = np.polyfit(w4,f4,3)

    continuum4 = p4[0]*w4**3 + p4[1]*w4**2 + p4[2]*w4 + p4[3]

    R4 = [(continuum4[i] - f4[i])/continuum4[i] for i in range(len(f4))]
    
    return R1, R2, R3, R4


def s_lambda_three_order(model, param, param_val,increment):    
    if param is 'teff':
        model.teff_0 = param_val
    elif param is 'logg':
        model.logg_0 = param_val
    elif param is 'mh':
        model.mh_0 = param_val

    R1_cen,R2_cen,R3_cen = r_val_polynomial_three_order(model)

    if param is 'teff':
        model.teff_0 = param_val+increment
    elif param is 'logg':
        model.logg_0 = param_val+increment
    elif param is 'mh':
        model.mh_0 = param_val+increment

    R1_up, R2_up, R3_up = r_val_polynomial_three_order(model)

    if param is 'teff':
        model.teff_0 = param_val-increment
    elif param is 'logg':
        model.logg_0 = param_val-increment
    elif param is 'mh':
        model.mh_0 = param_val-increment

    R1_dw,R2_dw,R3_dw = r_val_polynomial_three_order(model)
    
    s_lambda1 = [100*(R1_up[i] - R1_dw[i])/R1_cen[i] for i in range(len(R1_up))]

    s_lambda2 = [100*(R2_up[i] - R2_dw[i])/R2_cen[i] for i in range(len(R2_up))]

    s_lambda3 = [100*(R3_up[i] - R3_dw[i])/R3_cen[i] for i in range(len(R3_up))]

    return s_lambda1, s_lambda2, s_lambda3

def s_lambda_four_order(model, param, param_val,increment):
    
    if param is 'teff':
        model.teff_0 = param_val
    elif param is 'logg':
        model.logg_0 = param_val
    elif param is 'mh':
        model.mh_0 = param_val

    R1_cen,R2_cen,R3_cen,R4_cen = r_val_polynomial_four_order(model)

    if param is 'teff':
        model.teff_0 = param_val+increment
    elif param is 'logg':
        model.logg_0 = param_val+increment
    elif param is 'mh':
        model.mh_0 = param_val+increment

    R1_up, R2_up, R3_up,R4_up = r_val_polynomial_four_order(model)

    if param is 'teff':
        model.teff_0 = param_val-increment
    elif param is 'logg':
        model.logg_0 = param_val-increment
    elif param is 'mh':
        model.mh_0 = param_val-increment

    R1_dw,R2_dw,R3_dw,R4_dw = r_val_polynomial_four_order(model)
    
    s_lambda1 = [100*(R1_up[i] - R1_dw[i])/R1_cen[i] for i in range(len(R1_up))]

    s_lambda2 = [100*(R2_up[i] - R2_dw[i])/R2_cen[i] for i in range(len(R2_up))]

    s_lambda3 = [100*(R3_up[i] - R3_dw[i])/R3_cen[i] for i in range(len(R3_up))]
    
    s_lambda4 = [100*(R4_up[i] - R4_dw[i])/R4_cen[i] for i in range(len(R4_up))]

    return s_lambda1, s_lambda2, s_lambda3, s_lambda4



def load_full_grid():
    g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    return g


def plot_fit_result(starname,g,specdir='/group/data/nirspec/spectra/',
    fitpath='/u/rbentley/metallicity/spectra_fits',snr=30.0,nnorm=2):

    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')
    file4 = glob.glob(specdir+starname+'_order37*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='micron')
    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='micron')
    
    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]    
    waverange37 = [np.amin(starspectrum37.wavelength.value[:970]), np.amax(starspectrum37.wavelength.value[:970])]
    
    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
    
    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)
    starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum37.flux.unit

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

    interp4 = Interpolate(starspectrum37)
    convolve4 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot4 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm4 = Normalize(starspectrum37,nnorm)


    model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
         norm1 & norm2 & norm3 & norm4


    gc_result = MultiNestResult.from_hdf5(fitpath)

    for a in gc_result.median.keys():
        setattr(model,a,gc_result.median[a])

    
    w1,f1,w2,f2,w3,f3,w4,f4 = model()
    print model

    plt.plot(w1,f1)
    plt.plot(w2,f2)
    plt.plot(w3,f3)
    plt.plot(w4,f4)
    plt.plot(starspectrum34.wavelength.value,starspectrum34.flux.value)
    plt.plot(starspectrum35.wavelength.value,starspectrum35.flux.value)
    plt.plot(starspectrum36.wavelength.value,starspectrum36.flux.value)
    plt.plot(starspectrum37.wavelength.value,starspectrum37.flux.value)
    plt.show()
