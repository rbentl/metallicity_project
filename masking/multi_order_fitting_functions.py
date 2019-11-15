#from starkit.gridkit.io.phoenix.base import PhoenixSpectralGridIO, ParameterSet
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood
from starkit.fitkit.likelihoods import SpectralL1Likelihood as L1Likelihood
from starkit.fitkit.likelihoods import SpectralChi2LikelihoodAddErr as Chi2LikelihoodAddErr
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
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
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
from scipy import ndimage as nd

import model_tester_updated as mtu

try:
    import MySQLdb as mdb
except:
    import pymysql as mdb


#sns.set_context('paper',font_scale=2.0, rc={"lines.linewidth": 1.75})
#sns.set_style("white")
#sns.set_style('ticks')


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

    file1 = glob.glob(specdir + starname + '_order34*.dat')
    file2 = glob.glob(specdir + starname + '_order35*.dat')
    file3 = glob.glob(specdir + starname + '_order36*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                     wave_range=waverange34)
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum34.flux.unit

    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                     wave_range=waverange35)
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                     wave_range=waverange36)
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum36.flux.unit

    if savefile is None:
        savefile = os.path.join(savedir,'unmasked_'+starname+'_order34-36'+like_str+'.h5')
    # restore MultiNest savefile
    result = MultiNestResult.from_hdf5(savefile)

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
            for a in result.median.keys():
                setattr(model, a, result.median[a])

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
    residual_flux = (calc_residuals(f3,starspectrum36.flux.value),calc_residuals(f2,starspectrum35.flux.value),calc_residuals(f1,starspectrum34.flux.value))
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
        plt.plot(model_wave[i]*velfac,residual_flux[i])
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


def plot_fit_result_three_order(starname, fitpath, grid,
                               specdir='/group/data/nirspec/spectra/',
                               snr=30.0, nnorm=2, nirspec_upgrade=False, nsdrp_snr=False,triangle_plot=False):
    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        if nsdrp_snr is False:
            starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum34.flux.unit
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum35.flux.unit
            starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum36.flux.unit

    else:
        print specdir + starname + '_order34*.dat'
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]),
                       np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]),
                       np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]),
                       np.amax(starspectrum36.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(grid, R=24000)
    rot1 = RotationalBroadening.from_grid(grid, vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34, 2)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(grid, R=24000)
    norm2 = Normalize(starspectrum35, 2)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(grid, R=24000)
    norm3 = Normalize(starspectrum36, 2)


    model = grid | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
                 convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
                 norm1 & norm2 & norm3


    result = MultiNestResult.from_hdf5(fitpath)

    for a in result.median.keys():
        setattr(model, a, result.median[a])

    w1, f1, w2, f2, w3, f3 = model()

    sigmas = result.calculate_sigmas(1)

    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))

    if starname in [x[0] for x in cal_star_info]:
        print starname+' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]

    deltas = [result.median['teff_0']-cal_star_info[2], result.median['logg_0']-cal_star_info[3], result.median['mh_0']-cal_star_info[1], result.median['alpha_0']-cal_star_info[4]]

    res1 = starspectrum34.flux.value - f1

    res2 = starspectrum35.flux.value - f2

    res3 = starspectrum36.flux.value - f3

    chi2 = np.sum((res1)**2/(starspectrum34.uncertainty.value)**2)
    chi2 = chi2 + np.sum((res2)**2/(starspectrum35.uncertainty.value)**2)
    chi2 = chi2 + np.sum((res3)**2/(starspectrum36.uncertainty.value)**2)
    chi2  = chi2/(len(res1)+len(res2)+len(res3))

    print chi2

    if triangle_plot:
        result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0','vrot_1'])
        plt.show()

    plt.text(np.amax(starspectrum35.wavelength.value)-30,0.5,'APOGEE param offsets:\n$\Delta T_{eff}:$'+str(deltas[0])+'\n$\Delta log g:$'+str(deltas[1])+\
             '\n$\Delta [M/H]:$'+str(deltas[2])+'\n$\Delta alpha$:'+str(deltas[3])+'\nChi^2:'+str(chi2),fontsize=12)

    plt.plot(starspectrum34.wavelength.value / (result.median['vrad_3'] / 3e5 + 1.0), starspectrum34.flux.value,
             color='#FF0303', label='Data')
    plt.plot(starspectrum35.wavelength.value / (result.median['vrad_4'] / 3e5 + 1.0), starspectrum35.flux.value,
             color='#FF0303')
    plt.plot(starspectrum36.wavelength.value / (result.median['vrad_5'] / 3e5 + 1.0), starspectrum36.flux.value,
             color='#FF0303')


    plt.plot(w1 / (result.median['vrad_3'] / 3e5 + 1.0), f1, color='#F5A182', label='Model/Residuals')

    plt.plot(w1 / (result.median['vrad_3'] / 3e5 + 1.0), res1, color='#F5A182')

    plt.plot(w2 / (result.median['vrad_4'] / 3e5 + 1.0), f2, color='#F5A182')

    plt.plot(w2 / (result.median['vrad_4'] / 3e5 + 1.0), res2, color='#F5A182')

    plt.plot(w3 / (result.median['vrad_5'] / 3e5 + 1.0), f3, color='#F5A182')

    plt.plot(w3 / (result.median['vrad_5'] / 3e5 + 1.0), res3, color='#F5A182')

    plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
    plt.axhline(y=-0.05, color='k', linestyle='--')

    plt.legend(loc='upper left', fontsize=16)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Normalized Flux')
    plt.title(fitpath.split('/')[-1])
    plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=15)

    print result
    plt.show()


def fit_star_three_orders_sens_masked(starname,masking_param,g,model_type,sl_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                                     R_fixed=None,logg_fixed=None,nirspec_upgrade=False,nsdrp_snr=False,adderr=False):


    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
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
        starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        
        if nsdrp_snr is False:
            starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit
            starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    else:
        print specdir+starname+'_order34*.dat'
        file1 = glob.glob(specdir+starname+'_order34*.dat')
        file2 = glob.glob(specdir+starname+'_order35*.dat')
        file3 = glob.glob(specdir+starname+'_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom')

    
        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]), np.amax(starspectrum36.wavelength.value[:2000])]    
    
        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit

        
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
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

    
    if masking_param is 'mh':
        sl_mh1,sl_mh2,sl_mh3 = s_lambda_three_order(model,'mh',model.mh_0.value,0.1)
         
    elif masking_param is 'teff':
        sl_mh1,sl_mh2,sl_mh3 = s_lambda_three_order(model,'teff',model.teff_0.value,200)
         
    elif masking_param is 'logg':
        sl_mh1,sl_mh2,sl_mh3 = s_lambda_three_order(model,'logg',model.logg_0.value,0.1)
         
    elif masking_param is 'alpha':
        sl_mh1,sl_mh2,sl_mh3 = s_lambda_three_order(model,'alpha',model.alpha_0.value,0.1)

    w1,f1,w2,f2,w3,f3 = model()


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

    if adderr:
        sl_like1 = Chi2LikelihoodAddErr(masked_data_sl1)
        sl_like2 = Chi2LikelihoodAddErr(masked_data_sl2)
        sl_like3 = Chi2LikelihoodAddErr(masked_data_sl3)

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

    if adderr:
        add_err_prior1 = priors.UniformPrior(0,0.1)
        add_err_prior2 = priors.UniformPrior(0,0.1)
        add_err_prior3 = priors.UniformPrior(0,0.1)
    else:
        add_err_prior1 = priors.FixedPrior(0.)
        add_err_prior2 = priors.FixedPrior(0.)
        add_err_prior3 = priors.FixedPrior(0.)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, \
             vrad_prior1,vrad_prior2,vrad_prior3,R_prior1,R_prior2,\
             R_prior3,add_err_prior1,add_err_prior2,add_err_prior3])

    fitobj.run(verbose=verbose,importance_nested_sampling=False,n_live_points=400)
    result=fitobj.result

    if adderr:
        result.to_hdf(os.path.join(savedir,masking_param + '_masked_sl_cutoff_'+str(sl_cut)+'_'+starname+'_order34-36_'+model_type+'_adderr.h5'))
    else:
        result.to_hdf(os.path.join(savedir,masking_param + '_masked_sl_cutoff_'+str(sl_cut)+'_'+starname+'_order34-36_'+model_type+'.h5'))
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



def fit_star_four_orders_sens_masked(starname,masking_param,g,sl_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                                     R_fixed=None,logg_fixed=None,l1norm=False,nirspec_upgrade=False,nsdrp_snr=False):


    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
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
        waverange37 = [np.amin(starspectrum37.wavelength.value[600:970]), np.amax(starspectrum37.wavelength.value[:970])]
    
        starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)

        if nsdrp_snr is False:
            starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit
            starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
            starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum37.flux.unit

    else:
        print specdir+starname+'_order34*.dat'
        file1 = glob.glob(specdir+starname+'_order34*.dat')
        file2 = glob.glob(specdir+starname+'_order35*.dat')
        file3 = glob.glob(specdir+starname+'_order36*.dat')
        file4 = glob.glob(specdir+starname+'_order37*.dat')

        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom')
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom')

    
        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]), np.amax(starspectrum36.wavelength.value[:2000])]    
        waverange37 = [20800, np.amax(starspectrum37.wavelength.value[:2000])]
    
        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit

        
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
        
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)
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

    unmasked_result = MultiNestResult.from_hdf5(os.path.join(savedir,'masked_sl_cutoff_0.0_'+starname+'_order34-37.h5'))

    for a in unmasked_result.median.keys():
        setattr(model,a,unmasked_result.median[a])

    w1,f1,w2,f2,w3,f3,w4,f4 = model()
    print model

    if masking_param is 'mh':
        sl_mh1,sl_mh2,sl_mh3,sl_mh4 = s_lambda_four_order(model,'mh',model.mh_0.value,0.1)
         
    elif masking_param is 'teff':
        sl_mh1,sl_mh2,sl_mh3,sl_mh4 = s_lambda_four_order(model,'teff',model.teff_0.value,200)
         
    elif masking_param is 'logg':
        sl_mh1,sl_mh2,sl_mh3,sl_mh4 = s_lambda_four_order(model,'logg',model.logg_0.value,0.1)
         
    elif masking_param is 'alpha':
        sl_mh1,sl_mh2,sl_mh3,sl_mh4 = s_lambda_four_order(model,'alpha',model.alpha_0.value,0.1)

    else:
        print 'No mask selected'
        return
        
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
    result.to_hdf(os.path.join(savedir,masking_param + '_masked_sl_cutoff_'+str(sl_cut)+'_'+starname+'_order34-37'+like_str+'_bosz_order37trimmed.h5'))
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

    file1 = os.path.join(savedir,'textoutput/order34/'+masking_param+'_'+starname+'_order34_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file2 = os.path.join(savedir,'textoutput/order35/'+masking_param+'_'+starname+'_order35_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file3 = os.path.join(savedir,'textoutput/order36/'+masking_param+'_'+starname+'_order36_model_sl_cutoff_'+str(sl_cut)+'.txt')
    file4 = os.path.join(savedir,'textoutput/order37/'+masking_param+'_'+starname+'_order37_model_sl_cutoff_'+str(sl_cut)+'.txt')

    write_spectrum.write_txt(w1,f1,file1,comments=comment1)
    write_spectrum.write_txt(w2,f2,file2,comments=comment2)
    write_spectrum.write_txt(w3,f3,file3,comments=comment3)
    write_spectrum.write_txt(w4,f4,file4,comments=comment4)




def fit_star_one_order_sens_masked(starname,masking_param,g,sl_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                                     R_fixed=None,logg_fixed=None,l1norm=False,nirspec_upgrade=False,nsdrp_snr=False,vrot_fixed=None):


    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
        file1 = glob.glob(specdir+starname+'_order36*.dat')

        starspectrum35 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='micron')
    
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]

        starspectrum35 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange35)

        if nsdrp_snr is False:
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    else:
        file1 = glob.glob(specdir+starname+'_order36*.dat')

        starspectrum35 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
    
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]

        starspectrum35 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit
        
    interp1 = Interpolate(starspectrum35)
    convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
    rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum35,2)

    model = g | rot1 |DopplerShift(vrad=0.0)| convolve1 | interp1 | norm1

    # add likelihood parts
    like1 = Chi2Likelihood(starspectrum35)
    #like1_l1 = SpectralL1Likelihood(spectrum)

    fit_model = model | like1

    
    #unmasked_result = MultiNestResult.from_hdf5(os.path.join(savedir,'sensitivity_cut_0._NGC6791_J19205+3748282_order35.h5'))

    #for a in unmasked_result.median.keys():
    #    setattr(model,a,unmasked_result.median[a])

    w1,f1 = model()
    print model
    print w1, f1

    if masking_param is 'mh':
        sl_mh = mtu.s_lambda(model,'mh',model.mh_0.value,0.1)
         
    elif masking_param is 'teff':
        sl_mh = mtu.s_lambda(model,'teff',model.teff_0.value,200)
         
    elif masking_param is 'logg':
        sl_mh = mtu.s_lambda(model,'logg',model.logg_0.value,0.1)
         
    elif masking_param is 'alpha':
        sl_mh = mtu.s_lambda(model,'alpha',model.alpha_0.value,0.1)

    else:
        print 'No mask selected'
        return
        
    # likelihoods
    if l1norm:
        like = L1Likelihood(starspectrum35)

    else:
        like = Chi2Likelihood(starspectrum35)

    fit_model = model | like
    print fit_model.__class__
    print fit_model()

    mask_sl_f = []
    mask_sl_w = []
    sl_mask_indices = []

    for i in range(len(sl_mh)):
        if abs(sl_mh[i])<float(sl_cut):
            mask_sl_f += [starspectrum35.flux.value[i]]
            mask_sl_w += [starspectrum35.wavelength.value[i]]        
            sl_mask_indices += [i]
 
    masked_data_sl_f = np.delete(starspectrum35.flux.value,sl_mask_indices)
    masked_data_sl_w = np.delete(starspectrum35.wavelength.value,sl_mask_indices)
    masked_data_sl_u = np.delete(starspectrum35.uncertainty.value,sl_mask_indices)

    print "Number of included data points for one order:",len(masked_data_sl_f)
    masked_data_sl = Spectrum1D.from_array(dispersion=masked_data_sl_w, flux=masked_data_sl_f, dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u) #
    
    interp_sl = Interpolate(masked_data_sl)
    convolve_sl = InstrumentConvolveGrating.from_grid(g,R=24000)
    rot_sl = RotationalBroadening.from_grid(g,vrot=np.array([0.0]))
    norm_sl = Normalize(masked_data_sl,2)


    model = g | rot_sl |DopplerShift(vrad=0.0)| convolve_sl | interp_sl | norm_sl

    w1,f1 = model()

    if l1norm:
        sl_like = L1Likelihood(masked_data_sl)


    else:
        sl_like = Chi2Likelihood(masked_data_sl)


    fit_model = model | sl_like
    print fit_model.__class__
    print fit_model()
        

    teff_prior = priors.UniformPrior(*teff_range)
    if logg_fixed is not None:
        logg_prior = priors.FixedPrior(logg_fixed)
    else:
        logg_prior = priors.UniformPrior(*logg_range)
    mh_prior = priors.UniformPrior(*mh_range)
    alpha_prior = priors.UniformPrior(*alpha_range)

    if vrot_fixed is not None:
        vrot_prior = priors.FixedPrior(vrot_fixed)
    else: 
        vrot_prior = priors.UniformPrior(*vrot_range)
        
    vrad_prior1 = priors.UniformPrior(*vrad_range)

    # R_prior1 = priors.FixedPrior(R)
    # R_prior2 = priors.FixedPrior(R)
    # R_prior3 = priors.FixedPrior(R)
    # R_prior4 = priors.FixedPrior(R)

    if R_fixed is not None:
        R_prior1 = priors.FixedPrior(R_fixed)

    else:
        R_prior1 = priors.UniformPrior(*r_range)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior,vrot_prior, vrad_prior1,R_prior1])

    fitobj.run(verbose=verbose,importance_nested_sampling=False,n_live_points=400)
    result=fitobj.result

    if l1norm:
        like_str = '_l1norm'
    else:
        like_str = ''
    result.to_hdf(os.path.join(savedir,masking_param + '_masked_sl_cutoff_'+str(sl_cut)+'_'+starname+'_order36'+like_str+'_bosz.h5'))
    print result.calculate_sigmas(1)
    print result.median

    # save the individual model spectra with the max posterior value
    model.teff_0 = result.maximum['teff_0']
    model.logg_0 = result.maximum['logg_0']
    model.mh_0 = result.maximum['mh_0']
    model.alpha_0 = result.maximum['alpha_0']
    model.vrot_1 = result.maximum['vrot_1']
    model.vrad_2 = result.maximum['vrad_2']
    model.R_3 = result.maximum['R_3']


    w1,f1 = model()

    comment1 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value,model.logg_0.value,model.mh_0.value,model.alpha_0.value,
               model.vrot_1.value,model.vrad_2.value,model.R_3.value)

    #file1 = os.path.join(savedir,'textoutput/order35/'+masking_param+'_'+starname+'_order35_model_sl_cutoff_'+str(sl_cut)+'.txt')

    #write_spectrum.write_txt(w1,f1,file1,comments=comment1)


def fit_star_three_orders_convolved(starname, g, model_type, new_r,
                                      specdir='/group/data/nirspec/spectra/',
                                      savedir='../nirspec_fits/', snr=30.0, nnorm=2, teff_range=[2500, 6000],
                                      vrad_range=[-600, 600], logg_range=[0., 4.5], mh_range=[-2., 1.0],
                                      vrot_range=[0, 20],
                                      R=40000, verbose=True, alpha_range=[-1., 1.], r_range=[15000.0, 40000.0],
                                      R_fixed=None, logg_fixed=None, nirspec_upgrade=False, nsdrp_snr=False,
                                      adderr=False):
    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        if nsdrp_snr is False:
            starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum34.flux.unit
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum35.flux.unit
            starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum36.flux.unit

    else:
        print specdir + starname + '_order34*.dat'
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]),
                       np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]),
                       np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]),
                       np.amax(starspectrum36.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

    del_w1 = starspectrum34.wavelength.value[0]/(starspectrum34.wavelength.value[1]-starspectrum34.wavelength.value[0])
    sigma1 = del_w1/new_r/(2*np.sqrt(2*np.log(2)))
    conv_f1 = nd.gaussian_filter1d(starspectrum34.flux.value, sigma1)
    conv_spec34 = Spectrum1D.from_array(dispersion=starspectrum34.wavelength.value, flux=conv_f1, dispersion_unit=u.angstrom, uncertainty=starspectrum34.uncertainty.value)

    del_w2 = starspectrum35.wavelength.value[0]/(starspectrum35.wavelength.value[1]-starspectrum35.wavelength.value[0])
    sigma2 = del_w2/new_r/(2*np.sqrt(2*np.log(2)))
    conv_f2 = nd.gaussian_filter1d(starspectrum35.flux.value, sigma2)
    conv_spec35 = Spectrum1D.from_array(dispersion=starspectrum35.wavelength.value, flux=conv_f2, dispersion_unit=u.angstrom, uncertainty=starspectrum35.uncertainty.value)

    del_w3 = starspectrum36.wavelength.value[0]/(starspectrum36.wavelength.value[1]-starspectrum36.wavelength.value[0])
    sigma3 = del_w3/new_r/(2*np.sqrt(2*np.log(2)))
    conv_f3 = nd.gaussian_filter1d(starspectrum36.flux.value, sigma3)
    conv_spec36 = Spectrum1D.from_array(dispersion=starspectrum36.wavelength.value, flux=conv_f3, dispersion_unit=u.angstrom, uncertainty=starspectrum36.uncertainty.value)

    interp1 = Interpolate(conv_spec34)
    convolve1 = InstrumentConvolveGrating.from_grid(g, R=new_r)
    rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    norm1 = Normalize(conv_spec34, nnorm)

    interp2 = Interpolate(conv_spec35)
    convolve2 = InstrumentConvolveGrating.from_grid(g, R=new_r)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(conv_spec35, nnorm)

    interp3 = Interpolate(conv_spec36)
    convolve3 = InstrumentConvolveGrating.from_grid(g, R=new_r)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(conv_spec36, nnorm)

    model = g | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
            convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
            norm1 & norm2 & norm3

    w1, f1, w2, f2, w3, f3 = model()

    if adderr:
        like1 = Chi2LikelihoodAddErr(conv_spec34)
        like2 = Chi2LikelihoodAddErr(conv_spec35)
        like3 = Chi2LikelihoodAddErr(conv_spec36)

    else:
        like1 = Chi2Likelihood(conv_spec34)
        like2 = Chi2Likelihood(conv_spec35)
        like3 = Chi2Likelihood(conv_spec36)

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

    if adderr:
        add_err_prior1 = priors.UniformPrior(0, 0.1)
        add_err_prior2 = priors.UniformPrior(0, 0.1)
        add_err_prior3 = priors.UniformPrior(0, 0.1)
    else:
        add_err_prior1 = priors.FixedPrior(0.)
        add_err_prior2 = priors.FixedPrior(0.)
        add_err_prior3 = priors.FixedPrior(0.)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, \
                                   vrad_prior1, vrad_prior2, vrad_prior3, R_prior1, R_prior2, \
                                   R_prior3, add_err_prior1, add_err_prior2, add_err_prior3])

    fitobj.run(verbose=verbose, importance_nested_sampling=False, n_live_points=400)
    result = fitobj.result

    if adderr:
        result.to_hdf(os.path.join(savedir, 'convolved_'+str(new_r)+ '_' + starname + '_order34-36_' + model_type + '_adderr.h5'))
    else:
        result.to_hdf(os.path.join(savedir, 'convolved_'+str(new_r)+ '_' + starname + '_order34-36_' + model_type + '.h5'))
    print result.calculate_sigmas(1)
    print result.median


def fit_star_three_orders_fe_lines(starname, g, model_type,
                                      specdir='/group/data/nirspec/spectra/',
                                      savedir='../nirspec_fits/', snr=30.0, nnorm=2, teff_range=[2500, 6000],
                                      vrad_range=[-600, 600], logg_range=[0., 4.5], mh_range=[-2., 1.0],
                                      vrot_range=[0, 20],
                                      R=40000, verbose=True, alpha_range=[-1., 1.], r_range=[15000.0, 40000.0],
                                      R_fixed=None, logg_fixed=None, nirspec_upgrade=False, nsdrp_snr=False,
                                      adderr=False):
    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        if nsdrp_snr is False:
            starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum34.flux.unit
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum35.flux.unit
            starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum36.flux.unit

    else:
        print specdir + starname + '_order34*.dat'
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]),
                       np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]),
                       np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]),
                       np.amax(starspectrum36.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(g, R=24000)
    rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34, nnorm)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(starspectrum35, nnorm)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(starspectrum36, nnorm)

    model = g | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
            convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
            norm1 & norm2 & norm3

    w1, f1, w2, f2, w3, f3 = model()

    # starspectrumall_flux = np.concatenate((starspectrum34.flux.value,starspectrum35.flux.value,starspectrum36.flux.value))
    # starspectrumall_wavelength = np.concatenate((starspectrum34.wavelength.value,starspectrum35.wavelength.value,starspectrum36.wavelength.value))
    # starspectrumall_uncert = np.concatenate((starspectrum34.uncertainty.value,starspectrum35.uncertainty.value,starspectrum36.uncertainty.value))

    order36_line_index = np.where(np.logical_and(21200 <= starspectrum36.wavelength.value,starspectrum36.wavelength.value <= 21250))
    order36_line_index = np.append(order36_line_index,np.where(np.logical_and(21280 <= starspectrum36.wavelength.value,starspectrum36.wavelength.value <= 21300)))
    order36_line_index = np.append(order36_line_index,np.where(np.logical_and(21425 <= starspectrum36.wavelength.value,starspectrum36.wavelength.value <= 21440)))

    order35_line_index = np.where(np.logical_and(21795 <= starspectrum35.wavelength.value,starspectrum35.wavelength.value <= 21805))
    order35_line_index = np.append(order35_line_index,np.where(np.logical_and(21815 <= starspectrum35.wavelength.value,starspectrum35.wavelength.value <= 21825)))
    order35_line_index = np.append(order35_line_index,np.where(np.logical_and(21835 <= starspectrum35.wavelength.value,starspectrum35.wavelength.value <= 21885)))
    order35_line_index = np.append(order35_line_index,np.where(np.logical_and(21895 <= starspectrum35.wavelength.value,starspectrum35.wavelength.value <= 21905)))
    order35_line_index = np.append(order35_line_index,np.where(np.logical_and(21915 <= starspectrum35.wavelength.value,starspectrum35.wavelength.value <= 21925)))
    order35_line_index = np.append(order35_line_index,np.where(np.logical_and(22085 <= starspectrum35.wavelength.value,starspectrum35.wavelength.value <= 22905)))

    order34_line_index = np.where(np.logical_and(22460 <= starspectrum34.wavelength.value,starspectrum34.wavelength.value <= 22505))
    order34_line_index = np.append(order34_line_index,np.where(np.logical_and(22620 <= starspectrum34.wavelength.value,starspectrum34.wavelength.value <= 22630)))

    print order34_line_index,order35_line_index,order36_line_index

    masked_data_f1 = starspectrum34.flux.value[order34_line_index]
    masked_data_w1 = starspectrum34.wavelength.value[order34_line_index]
    masked_data_u1 = starspectrum34.uncertainty.value[order34_line_index]

    masked_data_f2 = starspectrum35.flux.value[order35_line_index]
    masked_data_w2 = starspectrum35.wavelength.value[order35_line_index]
    masked_data_u2 = starspectrum35.uncertainty.value[order35_line_index]

    masked_data_f3 = starspectrum36.flux.value[order36_line_index]
    masked_data_w3 = starspectrum36.wavelength.value[order36_line_index]
    masked_data_u3 = starspectrum36.uncertainty.value[order36_line_index]

    masked_data_1 = Spectrum1D.from_array(dispersion=masked_data_w1, flux=masked_data_f1,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_u1)
    masked_data_2 = Spectrum1D.from_array(dispersion=masked_data_w2, flux=masked_data_f2,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_u2)
    masked_data_3 = Spectrum1D.from_array(dispersion=masked_data_w3, flux=masked_data_f3,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_u3)

    interp1 = Interpolate(masked_data_1)
    convolve1 = InstrumentConvolveGrating.from_grid(g, R=24000)
    rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    norm1 = Normalize(masked_data_1, nnorm)

    interp2 = Interpolate(masked_data_2)
    convolve2 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(masked_data_2, nnorm)

    interp3 = Interpolate(masked_data_3)
    convolve3 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(masked_data_3, nnorm)

    model = g | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
            convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
            norm1 & norm2 & norm3

    w1, f1, w2, f2, w3, f3 = model()

    if adderr:
        like1 = Chi2LikelihoodAddErr(masked_data_1)
        like2 = Chi2LikelihoodAddErr(masked_data_2)
        like3 = Chi2LikelihoodAddErr(masked_data_3)

    else:
        like1 = Chi2Likelihood(masked_data_1)
        like2 = Chi2Likelihood(masked_data_2)
        like3 = Chi2Likelihood(masked_data_3)

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

    if adderr:
        add_err_prior1 = priors.UniformPrior(0, 0.1)
        add_err_prior2 = priors.UniformPrior(0, 0.1)
        add_err_prior3 = priors.UniformPrior(0, 0.1)
    else:
        add_err_prior1 = priors.FixedPrior(0.)
        add_err_prior2 = priors.FixedPrior(0.)
        add_err_prior3 = priors.FixedPrior(0.)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, \
                                   vrad_prior1, vrad_prior2, vrad_prior3, R_prior1, R_prior2, \
                                   R_prior3, add_err_prior1, add_err_prior2, add_err_prior3])

    fitobj.run(verbose=verbose, importance_nested_sampling=False, n_live_points=400)
    result = fitobj.result

    if adderr:
        result.to_hdf(os.path.join(savedir, 'masked_fe_lines_' + starname + '_order34-36_' + model_type + '_adderr.h5'))
    else:
        result.to_hdf(os.path.join(savedir, 'masked_fe_lines_' + starname + '_order34-36_' + model_type + '.h5'))
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

    w1, f1, w2, f2, w3, f3 = model()

    comment1 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value, model.logg_0.value, model.mh_0.value, model.alpha_0.value,
                model.vrot_1.value, model.vrad_3.value, model.R_6.value)
    comment2 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value, model.logg_0.value, model.mh_0.value, model.alpha_0.value,
                model.vrot_1.value, model.vrad_4.value, model.R_7.value)
    comment3 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value, model.logg_0.value, model.mh_0.value, model.alpha_0.value,
                model.vrot_1.value, model.vrad_5.value, model.R_8.value)


def load_full_grid_phoenix():
    g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    return g

def load_full_grid_bosz():
    g = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')
    return g


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
        if param_val-increment < model.teff_0.bounds[0]:
            param_val = model.teff_0.bounds[0]+increment       
        elif param_val+increment > model.teff_0.bounds[1]:
            param_val = model.teff_0.bounds[1]-increment
            
    elif param is 'logg':
        if param_val-increment < model.logg_0.bounds[0]:
            param_val = model.logg_0.bounds[0]+increment
        elif param_val+increment > model.logg_0.bounds[1]:
            param_val = model.logg_0.bounds[1]-increment
            
    elif param is 'mh':
        if param_val-increment < model.mh_0.bounds[0]:
            param_val = model.mh_0.bounds[0]+increment
        elif param_val+increment > model.mh_0.bounds[1]:
            param_val = model.mh_0.bounds[1]-increment
            
    elif param is 'alpha':
        if param_val-increment < model.alpha_0.bounds[0]:
            param_val = model.alpha_0.bounds[0]+increment
        elif param_val+increment > model.alpha_0.bounds[1]:
            param_val = model.alpha_0.bounds[1]-increment
    
    if param is 'teff':
        model.teff_0 = param_val
    elif param is 'logg':
        model.logg_0 = param_val
    elif param is 'mh':
        model.mh_0 = param_val
    elif param is 'alpha':
        model.alpha_0 = param_val

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
        if param_val-increment < model.teff_0.bounds[0]:
            param_val = model.teff_0.bounds[0]+increment       
        elif param_val+increment > model.teff_0.bounds[1]:
            param_val = model.teff_0.bounds[1]-increment
            
    elif param is 'logg':
        if param_val-increment < model.logg_0.bounds[0]:
            param_val = model.logg_0.bounds[0]+increment
        elif param_val+increment > model.logg_0.bounds[1]:
            param_val = model.logg_0.bounds[1]-increment
            
    elif param is 'mh':
        if param_val-increment < model.mh_0.bounds[0]:
            param_val = model.mh_0.bounds[0]+increment
        elif param_val+increment > model.mh_0.bounds[1]:
            param_val = model.mh_0.bounds[1]-increment
            
    elif param is 'alpha':
        if param_val-increment < model.alpha_0.bounds[0]:
            param_val = model.alpha_0.bounds[0]+increment
        elif param_val+increment > model.alpha_0.bounds[1]:
            param_val = model.alpha_0.bounds[1]-increment
    
    if param is 'teff':
        model.teff_0 = param_val
    elif param is 'logg':
        model.logg_0 = param_val
    elif param is 'mh':
        model.mh_0 = param_val
    elif param is 'alpha':
        model.alpha_0 = param_val

    R1_cen,R2_cen,R3_cen,R4_cen = r_val_polynomial_four_order(model)

    if param is 'teff':
        model.teff_0 = param_val+increment
    elif param is 'logg':
        model.logg_0 = param_val+increment
    elif param is 'mh':
        model.mh_0 = param_val+increment
    elif param is 'alpha':
        model.alpha_0 = param_val+increment

    R1_up, R2_up, R3_up,R4_up = r_val_polynomial_four_order(model)

    if param is 'teff':
        model.teff_0 = param_val-increment
    elif param is 'logg':
        model.logg_0 = param_val-increment
    elif param is 'mh':
        model.mh_0 = param_val-increment
    elif param is 'alpha':
        model.alpha_0 = param_val-increment

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
                    fitpath='/u/rbentley/metallicity/spectra_fits',snr=30.0,nnorm=2,nirspec_upgrade=False):

    if not nirspec_upgrade:
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

    else:
        print specdir+starname+'_order34*.dat'
        file1 = glob.glob(specdir+starname+'_order34*.dat')
        file2 = glob.glob(specdir+starname+'_order35*.dat')
        file3 = glob.glob(specdir+starname+'_order36*.dat')
        file4 = glob.glob(specdir+starname+'_order37*.dat')

        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom')
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]), np.amax(starspectrum36.wavelength.value[:2000])]    
        waverange37 = [20800, np.amax(starspectrum37.wavelength.value[:2000])]
    
        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
        
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
        
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)
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
        setattr(model, a, gc_result.median[a])


    w1,f1,w2,f2,w3,f3,w4,f4 = model()
    print model

    sigmas = gc_result.calculate_sigmas(1)

    print sigmas

    res1 = starspectrum34.flux.value-f1
    res2 = starspectrum35.flux.value-f2
    res3 = starspectrum36.flux.value-f3
    res4 = starspectrum37.flux.value-f4

    plt.plot(starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value, color='blue', label = 'Data')
    plt.plot(starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value, color='blue')
    plt.plot(starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value, color='blue')
    plt.plot(starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0),starspectrum37.flux.value, color='blue')

    plt.plot(w1/(gc_result.median['vrad_3']/3e5+1.0),f1, color='green', label = 'Model')
    plt.plot(w2/(gc_result.median['vrad_4']/3e5+1.0),f2, color='green')
    plt.plot(w3/(gc_result.median['vrad_5']/3e5+1.0),f3, color='green')
    plt.plot(w4/(gc_result.median['vrad_6']/3e5+1.0),f4, color='green')

    plt.plot(w1/(gc_result.median['vrad_3']/3e5+1.0),res1, color='red', label = 'Residuals')
    plt.plot(w2/(gc_result.median['vrad_4']/3e5+1.0),res2, color='red')
    plt.plot(w3/(gc_result.median['vrad_5']/3e5+1.0),res3, color='red')
    plt.plot(w4/(gc_result.median['vrad_6']/3e5+1.0),res4, color='red')

    plt.axhline(y=0.05, color='k', linestyle='--',label='$\pm$ 5%')
    plt.axhline(y=-0.05, color='k', linestyle='--')

    plt.legend()
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.title(fitpath.split('/')[-1])
    
    gc_result.plot_triangle(parameters=['teff_0','mh_0','logg_0','alpha_0','vrot_1'])
    plt.show()


def plot_fit_result_with_sl(starname, g, sl_cut, specdir='/group/data/nirspec/spectra/',
                    fitpath='/u/rbentley/metallicity/spectra_fits', snr=30.0, nnorm=2, nirspec_upgrade=False, masking_param='mh'):
    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')
        file4 = glob.glob(specdir + starname + '_order37*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')
        starspectrum37 = read_fits_file.read_nirspec_dat(file4, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]
        waverange37 = [np.amin(starspectrum37.wavelength.value[:970]), np.amax(starspectrum37.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        starspectrum37 = read_fits_file.read_nirspec_dat(file4, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange37)
        starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum37.flux.unit

    else:
        print specdir + starname + '_order34*.dat'
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')
        file4 = glob.glob(specdir + starname + '_order37*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom')
        starspectrum37 = read_nsdrp_txt(file4, desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]),
                       np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]),
                       np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]),
                       np.amax(starspectrum36.wavelength.value[:2000])]
        waverange37 = [20800, np.amax(starspectrum37.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        starspectrum37 = read_nsdrp_txt(file4, desired_wavelength_units='Angstrom', wave_range=waverange37)
        starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum37.flux.unit

    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(g, R=24000)
    rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34, nnorm)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(starspectrum35, nnorm)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(starspectrum36, nnorm)

    interp4 = Interpolate(starspectrum37)
    convolve4 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot4 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm4 = Normalize(starspectrum37, nnorm)

    model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(
        vrad=0) | \
            convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
            norm1 & norm2 & norm3 & norm4

    gc_result = MultiNestResult.from_hdf5(fitpath)

    if masking_param is 'mh':
        sl_mh1, sl_mh2, sl_mh3, sl_mh4 = s_lambda_four_order(model, 'mh', model.mh_0.value, 0.1)

    elif masking_param is 'teff':
        sl_mh1, sl_mh2, sl_mh3, sl_mh4 = s_lambda_four_order(model, 'teff', model.teff_0.value, 200)

    elif masking_param is 'logg':
        sl_mh1, sl_mh2, sl_mh3, sl_mh4 = s_lambda_four_order(model, 'logg', model.logg_0.value, 0.1)

    elif masking_param is 'alpha':
        sl_mh1, sl_mh2, sl_mh3, sl_mh4 = s_lambda_four_order(model, 'alpha', model.alpha_0.value, 0.1)

    for a in gc_result.median.keys():
        setattr(model, a, gc_result.median[a])

    sl_mask_indices1 = []
    sl_mask_indices2 = []
    sl_mask_indices3 = []
    sl_mask_indices4 = []

    for i in range(len(sl_mh1)):
        if abs(sl_mh1[i]) < float(sl_cut):
            sl_mask_indices1 += [i]

    for i in range(len(sl_mh2)):
        if abs(sl_mh2[i]) < float(sl_cut):
            sl_mask_indices2 += [i]

    for i in range(len(sl_mh3)):
        if abs(sl_mh3[i]) < float(sl_cut):
            print abs(sl_mh3[i])
            sl_mask_indices3 += [i]

    for i in range(len(sl_mh4)):
        if abs(sl_mh4[i]) < float(sl_cut):
            sl_mask_indices4 += [i]

    mask_w1 = starspectrum34.wavelength.value[sl_mask_indices1]
    mask_f1 = starspectrum34.flux.value[sl_mask_indices1]

    mask_w2 = starspectrum35.wavelength.value[sl_mask_indices2]
    mask_f2 = starspectrum35.flux.value[sl_mask_indices2]

    mask_w3 = starspectrum36.wavelength.value[sl_mask_indices2]
    mask_f3 = starspectrum36.flux.value[sl_mask_indices2]

    mask_w4 = starspectrum37.wavelength.value[sl_mask_indices2]
    mask_f4 = starspectrum37.flux.value[sl_mask_indices2]

    masked_data_sl_f1 = np.delete(starspectrum34.flux.value, sl_mask_indices1)
    masked_data_sl_w1 = np.delete(starspectrum34.wavelength.value, sl_mask_indices1)
    masked_data_sl_u1 = np.delete(starspectrum34.uncertainty.value, sl_mask_indices1)

    masked_data_sl_f2 = np.delete(starspectrum35.flux.value, sl_mask_indices2)
    masked_data_sl_w2 = np.delete(starspectrum35.wavelength.value, sl_mask_indices2)
    masked_data_sl_u2 = np.delete(starspectrum35.uncertainty.value, sl_mask_indices2)

    masked_data_sl_f3 = np.delete(starspectrum36.flux.value, sl_mask_indices3)
    masked_data_sl_w3 = np.delete(starspectrum36.wavelength.value, sl_mask_indices3)
    masked_data_sl_u3 = np.delete(starspectrum36.uncertainty.value, sl_mask_indices3)

    masked_data_sl_f4 = np.delete(starspectrum37.flux.value, sl_mask_indices4)
    masked_data_sl_w4 = np.delete(starspectrum37.wavelength.value, sl_mask_indices4)
    masked_data_sl_u4 = np.delete(starspectrum37.uncertainty.value, sl_mask_indices4)

    masked_data_sl1 = Spectrum1D.from_array(dispersion=masked_data_sl_w1, flux=masked_data_sl_f1,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u1)
    masked_data_sl2 = Spectrum1D.from_array(dispersion=masked_data_sl_w2, flux=masked_data_sl_f2,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u2)
    masked_data_sl3 = Spectrum1D.from_array(dispersion=masked_data_sl_w3, flux=masked_data_sl_f3,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u3)
    masked_data_sl4 = Spectrum1D.from_array(dispersion=masked_data_sl_w4, flux=masked_data_sl_f4,
                                            dispersion_unit=u.angstrom, uncertainty=masked_data_sl_u4)

    w1, f1, w2, f2, w3, f3, w4, f4 = model()
    print model

    sigmas = gc_result.calculate_sigmas(1)

    print sigmas

    res1 = starspectrum34.flux.value - f1
    res2 = starspectrum35.flux.value - f2
    res3 = starspectrum36.flux.value - f3
    res4 = starspectrum37.flux.value - f4

    fig, ax = plt.subplots()

    ax.plot(starspectrum34.wavelength.value / (gc_result.median['vrad_3'] / 3e5 + 1.0), starspectrum34.flux.value,
             color='blue', label='Data')
    ax.plot(starspectrum35.wavelength.value / (gc_result.median['vrad_4'] / 3e5 + 1.0), starspectrum35.flux.value,
             color='blue')
    ax.plot(starspectrum36.wavelength.value / (gc_result.median['vrad_5'] / 3e5 + 1.0), starspectrum36.flux.value,
             color='blue')
    ax.plot(starspectrum37.wavelength.value / (gc_result.median['vrad_6'] / 3e5 + 1.0), starspectrum37.flux.value,
             color='blue')
    '''

    plt.plot(masked_data_sl1.wavelength.value / (gc_result.median['vrad_3'] / 3e5 + 1.0), masked_data_sl1.flux.value,
             color='b', label='Unmasked Data')
    plt.plot(masked_data_sl2.wavelength.value / (gc_result.median['vrad_4'] / 3e5 + 1.0), masked_data_sl2.flux.value,
             color='b')
    plt.plot(masked_data_sl3.wavelength.value / (gc_result.median['vrad_5'] / 3e5 + 1.0), masked_data_sl3.flux.value,
             color='blue')
    plt.plot(masked_data_sl4.wavelength.value / (gc_result.median['vrad_6'] / 3e5 + 1.0), masked_data_sl4.flux.value,
             color='blue')

'   '''

    #plt.plot(w1 / (gc_result.median['vrad_3'] / 3e5 + 1.0), np.log10(sl_mh1), color='#DF1558', label='$S_{\lambda}$')
    #plt.plot(w2 / (gc_result.median['vrad_4'] / 3e5 + 1.0), np.log10(sl_mh2), color='#DF1558')
    #plt.plot(w3 / (gc_result.median['vrad_5'] / 3e5 + 1.0), np.log10(sl_mh3), color='#DF1558')
    #plt.plot(w4 / (gc_result.median['vrad_6'] / 3e5 + 1.0), np.log10(sl_mh4), color='#DF1558')

    #plt.plot(w1 / (gc_result.median['vrad_3'] / 3e5 + 1.0), f1, color='green', label='Model')
    #plt.plot(w2 / (gc_result.median['vrad_4'] / 3e5 + 1.0), f2, color='green')
    #plt.plot(w3 / (gc_result.median['vrad_5'] / 3e5 + 1.0), f3, color='green')
    #plt.plot(w4 / (gc_result.median['vrad_6'] / 3e5 + 1.0), f4, color='green')

    ax.plot(mask_w1 / (gc_result.median['vrad_3'] / 3e5 + 1.0), mask_f1,
             'r.', label='Masked Data')
    ax.plot(mask_w2 / (gc_result.median['vrad_4'] / 3e5 + 1.0), mask_f2,
             'r.')
    ax.plot(mask_w3 / (gc_result.median['vrad_5'] / 3e5 + 1.0), mask_f3,
             'r.')
    ax.plot(mask_w4 / (gc_result.median['vrad_6'] / 3e5 + 1.0), mask_f4,
             'r.')

    #ax.plot(w1 / (gc_result.median['vrad_3'] / 3e5 + 1.0), res1, color='red', label='Residuals')
    #ax.plot(w2 / (gc_result.median['vrad_4'] / 3e5 + 1.0), res2, color='red')
    #ax.plot(w3 / (gc_result.median['vrad_5'] / 3e5 + 1.0), res3, color='red')
    #ax.plot(w4 / (gc_result.median['vrad_6'] / 3e5 + 1.0), res4, color='red')

    #plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
    #plt.axhline(y=-0.05, color='k', linestyle='--')

    sl_ax = ax.twinx()

    abs_sl_mh2 = np.absolute(sl_mh2)

    sl_ax.plot(w1 / (gc_result.median['vrad_3'] / 3e5 + 1.0), np.log10(sl_mh1), color='#DF1558', label='Log (Sensitivity)')
    sl_ax.plot(w2 / (gc_result.median['vrad_4'] / 3e5 + 1.0), np.log10(abs_sl_mh2), color='#DF1558')
    sl_ax.plot(w3 / (gc_result.median['vrad_5'] / 3e5 + 1.0), np.log10(sl_mh3), color='#DF1558')
    sl_ax.plot(w4 / (gc_result.median['vrad_6'] / 3e5 + 1.0), np.log10(sl_mh4), color='#DF1558')

    sl_ax.axhline(y=np.log10(6.), color='k', linestyle='--', label='Sensitivity minimum cutoff')

    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('Flux')
    sl_ax.set_ylabel('Model Sensitivity')
    ax.set_ylim(-0.4, 1.4)
    sl_ax.set_ylim(-2.,7.)
    ax.legend(loc='upper left')
    sl_ax.legend(loc='center left')
    ax.set_title('Datapoints masked based on model sensitivity to [M/H] changes')

    plt.show()


def plot_fit_result_one_order(starname,g,specdir='/group/data/nirspec/spectra/',
                    fitpath='/u/rbentley/metallicity/spectra_fits',snr=30.0,nnorm=2,nirspec_upgrade=False):


    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
        file1 = glob.glob(specdir+starname+'_order35*.dat')

        starspectrum35 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='micron')
    
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]

        starspectrum35 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange35)

        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    else:
        file1 = glob.glob(specdir+starname+'_order35*.dat')

        starspectrum35 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
    
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]

        starspectrum35 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit
        
    interp1 = Interpolate(starspectrum35)
    convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
    rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum35,2)

    model = g | rot1 |DopplerShift(vrad=0.0)| convolve1 | interp1 | norm1


    gc_result = MultiNestResult.from_hdf5(fitpath)

    for a in gc_result.median.keys():
        setattr(model,a,gc_result.median[a])

    
    w1,f1 = model()
    print model

    sigmas = gc_result.calculate_sigmas(1)

    print sigmas

    res1 = starspectrum35.flux.value-f1

    high_residual_pts = 0
    
    for val in res1:
        if abs(val) > 0.05:
            high_residual_pts += 1

    plt.plot(starspectrum35.wavelength.value/(gc_result.median['vrad_2']/3e5+1.0),starspectrum35.flux.value, color='blue', label = 'Data')
 
    plt.plot(w1/(gc_result.median['vrad_2']/3e5+1.0),f1, color='green', label = 'Model')

    plt.plot(w1/(gc_result.median['vrad_2']/3e5+1.0),res1, color='red', label = 'Residuals')

    plt.axhline(y=0.05, color='k', linestyle='--',label='$\pm$ 5%')
    plt.axhline(y=-0.05, color='k', linestyle='--')

    plt.legend()
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.title(fitpath.split('/')[-1])
    
    gc_result.plot_triangle(parameters=['teff_0','mh_0','logg_0','alpha_0'])

    print "Number of points with residuals > 0.05:", high_residual_pts 
    
    plt.show()

    
def plot_all_residuals(starname,specdir='/u/rbentley/metallicity/spectra/',fitdir='/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/',\
                       snr=30.0,nnorm=2,nirspec_upgrade=False):
    if not nirspec_upgrade:
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

    else:
        file1 = glob.glob(specdir+starname+'_order34*.dat')
        file2 = glob.glob(specdir+starname+'_order35*.dat')
        file3 = glob.glob(specdir+starname+'_order36*.dat')
        file4 = glob.glob(specdir+starname+'_order37*.dat')

        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom')
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom')
    
        waverange34 = [np.amin(starspectrum34.wavelength.value[:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:2000]), np.amax(starspectrum36.wavelength.value[:2000])]    
        waverange37 = [np.amin(starspectrum37.wavelength.value[:2000]), np.amax(starspectrum37.wavelength.value[:2000])]
    
        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
        
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
        
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)
        starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum37.flux.unit


    g = load_full_grid()

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

    unmasked_result = MultiNestResult.from_hdf5(fitdir+'masked_sl_cutoff_0.0_'+starname+'_order34-37.h5')

    for a in unmasked_result.median.keys():
        setattr(model,a,unmasked_result.median[a])

    
    u_w1,u_f1,u_w2,u_f2,u_w3,u_f3,u_w4,u_f4 = model()

    unmasked_res1 = starspectrum34.flux.value-u_f1
    unmasked_res2 = starspectrum35.flux.value-u_f2
    unmasked_res3 = starspectrum36.flux.value-u_f3
    unmasked_res4 = starspectrum37.flux.value-u_f4

    if os.path.exists(fitdir+'masked_sl_cutoff_6.0_'+starname+'_order34-37.h5'):
        sl_masked_result = MultiNestResult.from_hdf5(fitdir+'masked_sl_cutoff_6.0_'+starname+'_order34-37.h5')
    elif os.path.exists(fitdir+'masked_sl_cutoff_6_'+starname+'_order34-37.h5'):
        sl_masked_result = MultiNestResult.from_hdf5(fitdir+'masked_sl_cutoff_6_'+starname+'_order34-37.h5')
    else:
        print 'No S_l mask'
        return

    for a in sl_masked_result.median.keys():
        setattr(model,a,sl_masked_result.median[a])

    
    w1,f1,w2,f2,w3,f3,w4,f4 = model()

    sl_res1 = starspectrum34.flux.value-f1
    sl_res2 = starspectrum35.flux.value-f2
    sl_res3 = starspectrum36.flux.value-f3
    sl_res4 = starspectrum37.flux.value-f4

    res_masked_result = MultiNestResult.from_hdf5(fitdir+'masked_res_cutoff_0.15_'+starname+'_order34-37.h5')

    for a in res_masked_result.median.keys():
        setattr(model,a,res_masked_result.median[a])

    
    w1,f1,w2,f2,w3,f3,w4,f4 = model()

    res_res1 = starspectrum34.flux.value-f1
    res_res2 = starspectrum35.flux.value-f2
    res_res3 = starspectrum36.flux.value-f3
    res_res4 = starspectrum37.flux.value-f4

    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))
    if starname in [x[0] for x in cal_star_info]:
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]
        setattr(model,'mh_0',cal_star_info[1])
        setattr(model,'teff_0',cal_star_info[2])
        setattr(model,'logg_0',cal_star_info[3])
        setattr(model,'alpha_0',cal_star_info[4])

    w1,f1,w2,f2,w3,f3,w4,f4 = model()

    ap_res1 = starspectrum34.flux.value-f1
    ap_res2 = starspectrum35.flux.value-f2
    ap_res3 = starspectrum36.flux.value-f3
    ap_res4 = starspectrum37.flux.value-f4

    plt.plot(w1/(unmasked_result.median['vrad_3']/3e5+1.0), unmasked_res1, color='blue',label='Unmasked Residual')
    plt.plot(w2/(unmasked_result.median['vrad_4']/3e5+1.0), unmasked_res2, color='blue')
    plt.plot(w3/(unmasked_result.median['vrad_5']/3e5+1.0), unmasked_res3, color='blue')
    plt.plot(w4/(unmasked_result.median['vrad_6']/3e5+1.0), unmasked_res4, color='blue')


    plt.plot(u_w1/(unmasked_result.median['vrad_3']/3e5+1.0), u_f1+1., color='green',label='Unmasked Model')
    plt.plot(u_w2/(unmasked_result.median['vrad_4']/3e5+1.0), u_f2+1., color='green')
    plt.plot(u_w3/(unmasked_result.median['vrad_5']/3e5+1.0), u_f3+1., color='green')
    plt.plot(u_w4/(unmasked_result.median['vrad_6']/3e5+1.0), u_f4+1., color='green')


    plt.axhline(y=0.05, color='k', linestyle='--',label='$\pm$ 5%')
    plt.axhline(y=0.0, color='k', label='0')
    plt.axhline(y=-0.05, color='k', linestyle='--')

    plt.plot(w1/(unmasked_result.median['vrad_3']/3e5+1.0), sl_res1+0.5, color='green',label='$S_{\lambda}$ Masked Residual')
    plt.plot(w2/(unmasked_result.median['vrad_4']/3e5+1.0), sl_res2+0.5, color='green')
    plt.plot(w3/(unmasked_result.median['vrad_5']/3e5+1.0), sl_res3+0.5, color='green')
    plt.plot(w4/(unmasked_result.median['vrad_6']/3e5+1.0), sl_res4+0.5, color='green')

    plt.axhline(y=0.5+0.05, color='k', linestyle='--')
    plt.axhline(y=0.0+0.5, color='k')
    plt.axhline(y=0.5-0.05, color='k', linestyle='--')
    
    plt.plot(w1/(unmasked_result.median['vrad_3']/3e5+1.0), res_res1+1., color='red',label='Residual Masked Residual')
    plt.plot(w2/(unmasked_result.median['vrad_4']/3e5+1.0), res_res2+1., color='red')
    plt.plot(w3/(unmasked_result.median['vrad_5']/3e5+1.0), res_res3+1., color='red')
    plt.plot(w4/(unmasked_result.median['vrad_6']/3e5+1.0), res_res4+1., color='red')

    plt.axhline(y=1.+0.05, color='k', linestyle='--')
    plt.axhline(y=0.0+1., color='k')
    plt.axhline(y=1.-0.05, color='k', linestyle='--')

    plt.plot(w1/(unmasked_result.median['vrad_3']/3e5+1.0), ap_res1+1.5, color='y',label='APOGEE Residual')
    plt.plot(w2/(unmasked_result.median['vrad_4']/3e5+1.0), ap_res2+1.5, color='y')
    plt.plot(w3/(unmasked_result.median['vrad_5']/3e5+1.0), ap_res3+1.5, color='y')
    plt.plot(w4/(unmasked_result.median['vrad_6']/3e5+1.0), ap_res4+1.5, color='y')
    
    plt.axhline(y=1.5+0.05, color='k', linestyle='--')
    plt.axhline(y=0.0+1.5, color='k')
    plt.axhline(y=1.5-0.05, color='k', linestyle='--')

    plt.plot(starspectrum34.wavelength.value/(unmasked_result.median['vrad_3']/3e5+1.0), starspectrum34.flux.value+1., color='k',label='Data')
    plt.plot(starspectrum35.wavelength.value/(unmasked_result.median['vrad_4']/3e5+1.0), starspectrum35.flux.value+1., color='k')
    plt.plot(starspectrum36.wavelength.value/(unmasked_result.median['vrad_5']/3e5+1.0), starspectrum36.flux.value+1., color='k')
    plt.plot(starspectrum37.wavelength.value/(unmasked_result.median['vrad_6']/3e5+1.0), starspectrum37.flux.value+1., color='k')
    
    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.25,molecules=False,size=12)
    plt.legend(loc='center left')
    plt.title(starname+' fitting residuals')
    plt.xlabel('Wavelength (Angstroms)')

    plt.show()


def plot_all_residuals_ksm(starname, specdir='/u/rbentley/metallicity/spectra/',
                       fitdir='/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/', \
                       snr=30.0, nnorm=2, nirspec_upgrade=False):
    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')
        file4 = glob.glob(specdir + starname + '_order37*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')
        starspectrum37 = read_fits_file.read_nirspec_dat(file4, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]
        waverange37 = [np.amin(starspectrum37.wavelength.value[:970]), np.amax(starspectrum37.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        starspectrum37 = read_fits_file.read_nirspec_dat(file4, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange37)
        starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum37.flux.unit

    else:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')
        file4 = glob.glob(specdir + starname + '_order37*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom')
        starspectrum37 = read_nsdrp_txt(file4, desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:2000]), np.amax(starspectrum36.wavelength.value[:2000])]
        waverange37 = [np.amin(starspectrum37.wavelength.value[:2000]), np.amax(starspectrum37.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        starspectrum37 = read_nsdrp_txt(file4, desired_wavelength_units='Angstrom', wave_range=waverange37)
        starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum37.flux.unit

    g = load_full_grid()

    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(g, R=24000)
    rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34, nnorm)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(starspectrum35, nnorm)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(starspectrum36, nnorm)

    interp4 = Interpolate(starspectrum37)
    convolve4 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot4 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm4 = Normalize(starspectrum37, nnorm)

    model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(
        vrad=0) | \
            convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
            norm1 & norm2 & norm3 & norm4

    unmasked_result = MultiNestResult.from_hdf5(fitdir + 'masked_sl_cutoff_0.0_' + starname + '_order34-37.h5')

    for a in unmasked_result.median.keys():
        setattr(model, a, unmasked_result.median[a])

    u_w1, u_f1, u_w2, u_f2, u_w3, u_f3, u_w4, u_f4 = model()

    unmasked_res1 = starspectrum34.flux.value - u_f1
    unmasked_res2 = starspectrum35.flux.value - u_f2
    unmasked_res3 = starspectrum36.flux.value - u_f3
    unmasked_res4 = starspectrum37.flux.value - u_f4

    if os.path.exists(fitdir + 'masked_sl_cutoff_6.0_' + starname + '_order34-37.h5'):
        sl_masked_result = MultiNestResult.from_hdf5(fitdir + 'masked_sl_cutoff_6.0_' + starname + '_order34-37.h5')
    elif os.path.exists(fitdir + 'masked_sl_cutoff_6_' + starname + '_order34-37.h5'):
        sl_masked_result = MultiNestResult.from_hdf5(fitdir + 'masked_sl_cutoff_6_' + starname + '_order34-37.h5')
    else:
        print 'No S_l mask'
        return

    for a in sl_masked_result.median.keys():
        setattr(model, a, sl_masked_result.median[a])

    w1, f1, w2, f2, w3, f3, w4, f4 = model()

    sl_res1 = starspectrum34.flux.value - f1
    sl_res2 = starspectrum35.flux.value - f2
    sl_res3 = starspectrum36.flux.value - f3
    sl_res4 = starspectrum37.flux.value - f4

    res_masked_result = MultiNestResult.from_hdf5(fitdir + 'masked_res_cutoff_0.15_' + starname + '_order34-37.h5')

    for a in res_masked_result.median.keys():
        setattr(model, a, res_masked_result.median[a])

    w1, f1, w2, f2, w3, f3, w4, f4 = model()

    res_res1 = starspectrum34.flux.value - f1
    res_res2 = starspectrum35.flux.value - f2
    res_res3 = starspectrum36.flux.value - f3
    res_res4 = starspectrum37.flux.value - f4

    cal_star_info = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    if starname in [x[0] for x in cal_star_info]:
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]
        setattr(model, 'mh_0', cal_star_info[1])
        setattr(model, 'teff_0', cal_star_info[2])
        setattr(model, 'logg_0', cal_star_info[3])
        setattr(model, 'alpha_0', cal_star_info[4])

    w1, f1, w2, f2, w3, f3, w4, f4 = model()

    ap_res1 = starspectrum34.flux.value - f1
    ap_res2 = starspectrum35.flux.value - f2
    ap_res3 = starspectrum36.flux.value - f3
    ap_res4 = starspectrum37.flux.value - f4

    plt.plot(w1 / (unmasked_result.median['vrad_3'] / 3e5 + 1.0), unmasked_res1, color='#F5A182',
             label='Unmasked Residual')
    plt.plot(w2 / (unmasked_result.median['vrad_4'] / 3e5 + 1.0), unmasked_res2, color='#F5A182')
    plt.plot(w3 / (unmasked_result.median['vrad_5'] / 3e5 + 1.0), unmasked_res3, color='#F5A182')
    plt.plot(w4 / (unmasked_result.median['vrad_6'] / 3e5 + 1.0), unmasked_res4, color='#F5A182')

    plt.plot(u_w1 / (unmasked_result.median['vrad_3'] / 3e5 + 1.0), u_f1 + 1., color='#F5A182', label='Unmasked Model')
    plt.plot(u_w2 / (unmasked_result.median['vrad_4'] / 3e5 + 1.0), u_f2 + 1., color='#F5A182')
    plt.plot(u_w3 / (unmasked_result.median['vrad_5'] / 3e5 + 1.0), u_f3 + 1., color='#F5A182')
    plt.plot(u_w4 / (unmasked_result.median['vrad_6'] / 3e5 + 1.0), u_f4 + 1., color='#F5A182')

    plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
    plt.axhline(y=0.0, color='k', label='0')
    plt.axhline(y=-0.05, color='k', linestyle='--')

    plt.plot(w1 / (unmasked_result.median['vrad_3'] / 3e5 + 1.0), sl_res1 + 0.5, color='green',
             label='$S_{\lambda}$ Masked Residual')
    plt.plot(w2 / (unmasked_result.median['vrad_4'] / 3e5 + 1.0), sl_res2 + 0.5, color='green')
    plt.plot(w3 / (unmasked_result.median['vrad_5'] / 3e5 + 1.0), sl_res3 + 0.5, color='green')
    plt.plot(w4 / (unmasked_result.median['vrad_6'] / 3e5 + 1.0), sl_res4 + 0.5, color='green')

    plt.axhline(y=0.5 + 0.05, color='k', linestyle='--')
    plt.axhline(y=0.5 - 0.05, color='k', linestyle='--')

    plt.plot(w1 / (unmasked_result.median['vrad_3'] / 3e5 + 1.0), res_res1 + 1., color='red',
             label='Residual Masked Residual')
    plt.plot(w2 / (unmasked_result.median['vrad_4'] / 3e5 + 1.0), res_res2 + 1., color='red')
    plt.plot(w3 / (unmasked_result.median['vrad_5'] / 3e5 + 1.0), res_res3 + 1., color='red')
    plt.plot(w4 / (unmasked_result.median['vrad_6'] / 3e5 + 1.0), res_res4 + 1., color='red')

    plt.axhline(y=1. + 0.05, color='k', linestyle='--')
    plt.axhline(y=1. - 0.05, color='k', linestyle='--')

    plt.plot(starspectrum34.wavelength.value / (unmasked_result.median['vrad_3'] / 3e5 + 1.0),
             starspectrum34.flux.value + 1., color='#FF0303', label='Data')
    plt.plot(starspectrum35.wavelength.value / (unmasked_result.median['vrad_4'] / 3e5 + 1.0),
             starspectrum35.flux.value + 1., color='#FF0303')
    plt.plot(starspectrum36.wavelength.value / (unmasked_result.median['vrad_5'] / 3e5 + 1.0),
             starspectrum36.flux.value + 1., color='#FF0303')
    plt.plot(starspectrum37.wavelength.value / (unmasked_result.median['vrad_6'] / 3e5 + 1.0),
             starspectrum37.flux.value + 1., color='#FF0303')

    plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=12)
    plt.legend(loc='center left')
    plt.title(starname + ' fitting residuals')
    plt.xlabel('Wavelength (Angstroms)')

    plt.show()


def fit_star_four_orders_residual_masked(starname,g,res_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                                         R_fixed=None,logg_fixed=None,l1norm=False,nirspec_upgrade=False):


    # fit a spectrum of a star with multiple orders that can have different velocities

    if not nirspec_upgrade:
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

    else:
        print specdir+starname+'_order34*.dat'
        file1 = glob.glob(specdir+starname+'_order34*.dat')
        file2 = glob.glob(specdir+starname+'_order35*.dat')
        file3 = glob.glob(specdir+starname+'_order36*.dat')
        file4 = glob.glob(specdir+starname+'_order37*.dat')

        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='micron')
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='micron')
        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='micron')
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='micron')
    
        waverange34 = [np.amin(starspectrum34.wavelength.value[:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:2000]), np.amax(starspectrum36.wavelength.value[:2000])]    
        waverange37 = [np.amin(starspectrum37.wavelength.value[:2000]), np.amax(starspectrum37.wavelength.value[:2000])]
    
        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
        
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
        
        starspectrum37 = read_nsdrp_txt(file4,desired_wavelength_units='Angstrom',wave_range=waverange37)
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


    if os.path.exists(os.path.join(savedir,'masked_sl_cutoff_0.0_'+starname+'_order34-37.h5')):
        unmasked_result = MultiNestResult.from_hdf5(os.path.join(savedir,'masked_sl_cutoff_0.0_'+starname+'_order34-37.h5'))
        print 'Setting model to unmasked values...'
        for a in unmasked_result.median.keys():
            setattr(model,a,unmasked_result.median[a])



    w1,f1,w2,f2,w3,f3,w4,f4 = model()
    print model


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


    residual_flux34 = calc_residuals(f1,starspectrum34.flux.value)
    residual_flux35 = calc_residuals(f2,starspectrum35.flux.value)
    residual_flux36 = calc_residuals(f3,starspectrum36.flux.value)
    residual_flux37 = calc_residuals(f4,starspectrum37.flux.value)

    res_masked_flux34, res_masked_wavelength34, res_masked_uncert34 = mask_data_with_residuals(starspectrum34.flux.value,starspectrum34.wavelength.value,starspectrum34.uncertainty.value,residual_flux34,res_cut)
    res_masked_flux35, res_masked_wavelength35, res_masked_uncert35 = mask_data_with_residuals(starspectrum35.flux.value,starspectrum35.wavelength.value,starspectrum35.uncertainty.value,residual_flux35,res_cut)
    res_masked_flux36, res_masked_wavelength36, res_masked_uncert36 = mask_data_with_residuals(starspectrum36.flux.value,starspectrum36.wavelength.value,starspectrum36.uncertainty.value,residual_flux36,res_cut)
    res_masked_flux37, res_masked_wavelength37, res_masked_uncert37 = mask_data_with_residuals(starspectrum37.flux.value,starspectrum37.wavelength.value,starspectrum37.uncertainty.value,residual_flux37,res_cut)



    masked_data_res1 = Spectrum1D.from_array(dispersion=res_masked_wavelength34, flux=res_masked_flux34, dispersion_unit=u.angstrom, uncertainty=res_masked_uncert34)
    masked_data_res2 = Spectrum1D.from_array(dispersion=res_masked_wavelength35, flux=res_masked_flux35, dispersion_unit=u.angstrom, uncertainty=res_masked_uncert35)
    masked_data_res3 = Spectrum1D.from_array(dispersion=res_masked_wavelength36, flux=res_masked_flux36, dispersion_unit=u.angstrom, uncertainty=res_masked_uncert36)
    masked_data_res4 = Spectrum1D.from_array(dispersion=res_masked_wavelength37, flux=res_masked_flux37, dispersion_unit=u.angstrom, uncertainty=res_masked_uncert37)
    
    mask_interp1 = Interpolate(masked_data_res1)
    mask_convolve1 = InstrumentConvolveGrating.from_grid(g,R=24000)
    mask_rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    mask_norm1 = Normalize(masked_data_res1,nnorm)

    mask_interp2 = Interpolate(masked_data_res2)
    mask_convolve2 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    mask_norm2 = Normalize(masked_data_res2,nnorm)

    mask_interp3 = Interpolate(masked_data_res3)
    mask_convolve3 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    mask_norm3 = Normalize(masked_data_res3,nnorm)

    
    mask_interp4 = Interpolate(masked_data_res4)
    mask_convolve4 = InstrumentConvolveGrating.from_grid(g,R=24000)
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    mask_norm4 = Normalize(masked_data_res4,nnorm)


    model = g | mask_rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         mask_convolve1 & mask_convolve2 & mask_convolve3 & mask_convolve4 | mask_interp1 & mask_interp2 & mask_interp3 & mask_interp4 | \
         mask_norm1 & mask_norm2 & mask_norm3 & mask_norm4

    w1,f1,w2,f2,w3,f3,w4,f4 = model()

    if l1norm:
        mask_like1 = L1Likelihood(masked_data_res1)
        mask_like2 = L1Likelihood(masked_data_res2)
        mask_like3 = L1Likelihood(masked_data_res3)
        mask_like4 = L1Likelihood(masked_data_res4)

    else:
        mask_like1 = Chi2Likelihood(masked_data_res1)
        mask_like2 = Chi2Likelihood(masked_data_res2)
        mask_like3 = Chi2Likelihood(masked_data_res3)
        mask_like4 = Chi2Likelihood(masked_data_res4)

    fit_model = model | mask_like1 & mask_like2 & mask_like3 & mask_like4 | Combiner4()
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
    result.to_hdf(os.path.join(savedir,'masked_res_cutoff_'+str(res_cut)+'_'+starname+'_order34-37'+like_str+'.h5'))
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

    file1 = os.path.join(savedir,'textoutput/order34/'+starname+'_order34_model_res_cutoff_'+str(res_cut)+'.txt')
    file2 = os.path.join(savedir,'textoutput/order35/'+starname+'_order35_model_res_cutoff_'+str(res_cut)+'.txt')
    file3 = os.path.join(savedir,'textoutput/order36/'+starname+'_order36_model_res_cutoff_'+str(res_cut)+'.txt')
    file4 = os.path.join(savedir,'textoutput/order37/'+starname+'_order37_model_res_cutoff_'+str(res_cut)+'.txt')

    write_spectrum.write_txt(w1,f1,file1,comments=comment1)
    write_spectrum.write_txt(w2,f2,file2,comments=comment2)
    write_spectrum.write_txt(w3,f3,file3,comments=comment3)
    write_spectrum.write_txt(w4,f4,file4,comments=comment4)


def fit_star_three_orders_residual_masked(starname,g,model_type,res_cut,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[2500,6000],
    vrad_range=[-600,600],logg_range=[0.,4.5],mh_range=[-2.,1.0],vrot_range=[0,20],
    R=40000,verbose=True,alpha_range=[-1.,1.],r_range=[15000.0,40000.0],
                                     R_fixed=None,logg_fixed=None,l1norm=False,nirspec_upgrade=False,nsdrp_snr=False):
    # fit a spectrum of a star with multiple orders that can have different velocities

    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

    else:
        print specdir + starname + '_order34*.dat'
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='micron')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='micron')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:2000]), np.amax(starspectrum36.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit


    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(g, R=24000)
    rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34, nnorm)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm2 = Normalize(starspectrum35, nnorm)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm3 = Normalize(starspectrum36, nnorm)

    model = g | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
            convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
            norm1 & norm2 & norm3

    if os.path.exists(os.path.join(savedir, 'masked_sl_cutoff_0.0_' + starname + '_order34-37.h5')):
        unmasked_result = MultiNestResult.from_hdf5(
            os.path.join(savedir, 'mh_masked_sl_cutoff_0.0_' + starname + '_order34-36.h5'))
        print 'Setting model to unmasked values...'
        for a in unmasked_result.median.keys():
            setattr(model, a, unmasked_result.median[a])

    w1, f1, w2, f2, w3, f3 = model()
    print model

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

    residual_flux34 = calc_residuals(f1, starspectrum34.flux.value)
    residual_flux35 = calc_residuals(f2, starspectrum35.flux.value)
    residual_flux36 = calc_residuals(f3, starspectrum36.flux.value)

    res_masked_flux34, res_masked_wavelength34, res_masked_uncert34 = mask_data_with_residuals(
        starspectrum34.flux.value, starspectrum34.wavelength.value, starspectrum34.uncertainty.value, residual_flux34,
        res_cut)
    res_masked_flux35, res_masked_wavelength35, res_masked_uncert35 = mask_data_with_residuals(
        starspectrum35.flux.value, starspectrum35.wavelength.value, starspectrum35.uncertainty.value, residual_flux35,
        res_cut)
    res_masked_flux36, res_masked_wavelength36, res_masked_uncert36 = mask_data_with_residuals(
        starspectrum36.flux.value, starspectrum36.wavelength.value, starspectrum36.uncertainty.value, residual_flux36,
        res_cut)

    masked_data_res1 = Spectrum1D.from_array(dispersion=res_masked_wavelength34, flux=res_masked_flux34,
                                             dispersion_unit=u.angstrom, uncertainty=res_masked_uncert34)
    masked_data_res2 = Spectrum1D.from_array(dispersion=res_masked_wavelength35, flux=res_masked_flux35,
                                             dispersion_unit=u.angstrom, uncertainty=res_masked_uncert35)
    masked_data_res3 = Spectrum1D.from_array(dispersion=res_masked_wavelength36, flux=res_masked_flux36,
                                             dispersion_unit=u.angstrom, uncertainty=res_masked_uncert36)

    mask_interp1 = Interpolate(masked_data_res1)
    mask_convolve1 = InstrumentConvolveGrating.from_grid(g, R=24000)
    mask_rot1 = RotationalBroadening.from_grid(g, vrot=np.array([10.0]))
    mask_norm1 = Normalize(masked_data_res1, nnorm)

    mask_interp2 = Interpolate(masked_data_res2)
    mask_convolve2 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    mask_norm2 = Normalize(masked_data_res2, nnorm)

    mask_interp3 = Interpolate(masked_data_res3)
    mask_convolve3 = InstrumentConvolveGrating.from_grid(g, R=24000)
    # rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    mask_norm3 = Normalize(masked_data_res3, nnorm)

    model = g | mask_rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
            mask_convolve1 & mask_convolve2 & mask_convolve3 | mask_interp1 & mask_interp2 & mask_interp3 | \
            mask_norm1 & mask_norm2 & mask_norm3

    w1, f1, w2, f2, w3, f3 = model()

    if l1norm:
        mask_like1 = L1Likelihood(masked_data_res1)
        mask_like2 = L1Likelihood(masked_data_res2)
        mask_like3 = L1Likelihood(masked_data_res3)

    else:
        mask_like1 = Chi2Likelihood(masked_data_res1)
        mask_like2 = Chi2Likelihood(masked_data_res2)
        mask_like3 = Chi2Likelihood(masked_data_res3)

    fit_model = model | mask_like1 & mask_like2 & mask_like3 | Combiner3()
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

    if R_fixed is not None:
        R_prior1 = priors.FixedPrior(R_fixed)
        R_prior2 = priors.FixedPrior(R_fixed)
        R_prior3 = priors.FixedPrior(R_fixed)
    else:
        R_prior1 = priors.UniformPrior(*r_range)
        R_prior2 = priors.UniformPrior(*r_range)
        R_prior3 = priors.UniformPrior(*r_range)

    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, \
                                   vrad_prior1, vrad_prior2, vrad_prior3, R_prior1, R_prior2, \
                                   R_prior3])

    fitobj.run(verbose=verbose, importance_nested_sampling=False, n_live_points=400)
    result = fitobj.result

    result.to_hdf(
        os.path.join(savedir, 'masked_res_cutoff_' + str(res_cut) + '_' + starname + '_order34-36_' + model_type +'.h5'))
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

    w1, f1, w2, f2, w3, f3 = model()

    comment1 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value, model.logg_0.value, model.mh_0.value, model.alpha_0.value,
                model.vrot_1.value, model.vrad_3.value, model.R_6.value)
    comment2 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value, model.logg_0.value, model.mh_0.value, model.alpha_0.value,
                model.vrot_1.value, model.vrad_4.value, model.R_7.value)
    comment3 = 'teff %f,logg %f,mh %f,alpha %f,vrot %f,vrad %f,R %f' % \
               (model.teff_0.value, model.logg_0.value, model.mh_0.value, model.alpha_0.value,
                model.vrot_1.value, model.vrad_5.value, model.R_8.value)

    file1 = os.path.join(savedir,
                         'textoutput/order34/' + starname + '_order34_model_res_cutoff_' + str(res_cut) + '.txt')
    file2 = os.path.join(savedir,
                         'textoutput/order35/' + starname + '_order35_model_res_cutoff_' + str(res_cut) + '.txt')
    file3 = os.path.join(savedir,
                         'textoutput/order36/' + starname + '_order36_model_res_cutoff_' + str(res_cut) + '.txt')

    write_spectrum.write_txt(w1, f1, file1, comments=comment1)
    write_spectrum.write_txt(w2, f2, file2, comments=comment2)
    write_spectrum.write_txt(w3, f3, file3, comments=comment3)

def compare_models_one_order(starname,boszpath,phoenixpath,bosz_grid,phoenix_grid,specdir='/group/data/nirspec/spectra/',
                    snr=30.0,nnorm=2,nirspec_upgrade=False):


    # fit a spectrum of a star with multiple orders that can have different velocities
    if not nirspec_upgrade:
        file1 = glob.glob(specdir+starname+'_order34*.dat')

        starspectrum35 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='micron')
    
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]

        starspectrum35 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=waverange35)

        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    else:
        file1 = glob.glob(specdir+starname+'_order34*.dat')

        starspectrum35 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
    
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]

        starspectrum35 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit
        
    interpb = Interpolate(starspectrum35)
    convolveb = InstrumentConvolveGrating.from_grid(bosz_grid,R=24000)
    rotb = RotationalBroadening.from_grid(bosz_grid,vrot=np.array([10.0]))
    normb = Normalize(starspectrum35,2)

    interpp = Interpolate(starspectrum35)
    convolvep = InstrumentConvolveGrating.from_grid(phoenix_grid,R=24000)
    rotp = RotationalBroadening.from_grid(phoenix_grid,vrot=np.array([10.0]))
    normp = Normalize(starspectrum35,2)

    bosz_model = bosz_grid | rotb |DopplerShift(vrad=0.0)| convolveb | interpb | normb

    phoenix_model = phoenix_grid | rotp |DopplerShift(vrad=0.0)| convolvep | interpp | normp


    bosz_result = MultiNestResult.from_hdf5(boszpath)

    phoenix_result = MultiNestResult.from_hdf5(phoenixpath)

    for a in bosz_result.median.keys():
        setattr(bosz_model,a,bosz_result.median[a])

    for a in phoenix_result.median.keys():
        setattr(phoenix_model,a,phoenix_result.median[a])

    
    bw1,bf1 = bosz_model()

    pw1,pf1 = phoenix_model()
    

    bsigmas = bosz_result.calculate_sigmas(1)
    psigmas = phoenix_result.calculate_sigmas(1)


    bosz_res1 = starspectrum35.flux.value-bf1
    phoenix_res1 = starspectrum35.flux.value-pf1

    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))

    if starname in [x[0] for x in cal_star_info]:
        print starname+' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]


    bosz_model.teff_0 = cal_star_info[2]
    bosz_model.logg_0 = cal_star_info[3]
    bosz_model.logg_0 = cal_star_info[1]
    bosz_model.alpha_0 = cal_star_info[4]
    ap_bw1,ap_bf1 = bosz_model()


    phoenix_model.teff_0 = cal_star_info[2]
    phoenix_model.logg_0 = cal_star_info[3]
    phoenix_model.logg_0 = cal_star_info[1]
    phoenix_model.alpha_0 = cal_star_info[4]
    ap_pw1,ap_pf1 = phoenix_model()
    
    bosz_ap_res1 = starspectrum35.flux.value-ap_bf1
    phoenix_ap_res1 = starspectrum35.flux.value-ap_pf1  

    bosz_deltas = [bosz_result.median['teff_0']-cal_star_info[2], bosz_result.median['logg_0']-cal_star_info[3], bosz_result.median['mh_0']-cal_star_info[1], bosz_result.median['alpha_0']-cal_star_info[4]]
    phoenix_deltas = [phoenix_result.median['teff_0']-cal_star_info[2], phoenix_result.median['logg_0']-cal_star_info[3], phoenix_result.median['mh_0']-cal_star_info[1], phoenix_result.median['alpha_0']-cal_star_info[4]]

    plt.text(np.amax(starspectrum35.wavelength.value)-30,1.03,'PHOENIX-APOGEE param offsets:\n$\Delta T_{eff}:$'+str(phoenix_deltas[0])+'\n$\Delta log g:$'+str(phoenix_deltas[1])+\
             '\n$\Delta [M/H]:$'+str(phoenix_deltas[2])+'\n$\Delta alpha$:'+str(phoenix_deltas[3])+'\nResidual StDev:'+str(np.std(phoenix_res1)),fontsize=12)

    plt.text(np.amax(starspectrum35.wavelength.value)-30,0.03,'BOSZ-APOGEE param offsets:\n$\Delta T_{eff}:$'+str(bosz_deltas[0])+'\n$\Delta log g:$'+str(bosz_deltas[1])+'\n$\Delta [M/H]:$'+\
             str(bosz_deltas[2])+'\n$\Delta alpha$:'+str(bosz_deltas[3])+'\nResidual StDev:'+str(np.std(bosz_res1)),fontsize=12)
    
    plt.plot(starspectrum35.wavelength.value/(phoenix_result.median['vrad_2']/3e5+1.0),starspectrum35.flux.value, color='#FF0303', label = 'Data')

    plt.plot(starspectrum35.wavelength.value/(bosz_result.median['vrad_2']/3e5+1.0),starspectrum35.flux.value+0.5, color='#0394FF', label = 'Data')


    plt.plot(pw1/(phoenix_result.median['vrad_2']/3e5+1.0),pf1, color='#F5A182', label = 'PHOENIX Model/Residuals')

    plt.plot(pw1/(phoenix_result.median['vrad_2']/3e5+1.0),phoenix_res1-0., color='#F5A182')
    
 
    plt.plot(bw1/(bosz_result.median['vrad_2']/3e5+1.0),bf1+0.5, color='#52D1DE', label = 'BOSZ Model/Residuals')

    plt.plot(bw1/(bosz_result.median['vrad_2']/3e5+1.0),bosz_res1+0.5, color='#52D1DE')


    #plt.plot(ap_pw1/(phoenix_result.median['vrad_2']/3e5+1.0),phoenix_ap_res1-1.0, color='#DF1558', label = 'PHOENIX Residuals at APOGEE values')

    #plt.plot(ap_bw1/(bosz_result.median['vrad_2']/3e5+1.0),bosz_ap_res1-0.5, color='#5867E5', label = 'BOSZ Residuals at APOGEE values')



    plt.axhline(y=0.5+0.05, color='k', linestyle='--',label='$\pm$ 5%')
    plt.axhline(y=0.5-0.05, color='k', linestyle='--')

    plt.axhline(y=-0.+0.05, color='k', linestyle='--')
    plt.axhline(y=-0.-0.05, color='k', linestyle='--')

    #plt.axhline(y=-0.5+0.05, color='k', linestyle='--')
    #plt.axhline(y=-0.5-0.05, color='k', linestyle='--')

    #plt.axhline(y=-1.0+0.05, color='k', linestyle='--')
    #plt.axhline(y=-1.0-0.05, color='k', linestyle='--')

    plt.legend(loc='lower right',fontsize=12)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.title(starname+' order 34 PHOENIX and BOSZ fits')
    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.25,molecules=False,size=12)
    
    #bosz_result.plot_triangle(parameters=['teff_0','mh_0','logg_0','alpha_0'])
    
    #phoenix_result.plot_triangle(parameters=['teff_0','mh_0','logg_0','alpha_0'])
    print bosz_result
    print phoenix_result
    plt.show()


def compare_models_three_order(starname,boszpath,phoenixpath,bosz_grid,phoenix_grid,specdir='/group/data/nirspec/spectra/',
                               snr=30.0,nnorm=2,nirspec_upgrade=False,nsdrp_snr=False):
    if not nirspec_upgrade:
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
        starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        
        if nsdrp_snr is False:
            starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
            starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit
            starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    else:
        print specdir+starname+'_order34*.dat'
        file1 = glob.glob(specdir+starname+'_order34*.dat')
        file2 = glob.glob(specdir+starname+'_order35*.dat')
        file3 = glob.glob(specdir+starname+'_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom')

    
        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]), np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]), np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]), np.amax(starspectrum36.wavelength.value[:2000])]    
    
        starspectrum34 = read_nsdrp_txt(file1,desired_wavelength_units='Angstrom',wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit

        
        starspectrum35 = read_nsdrp_txt(file2,desired_wavelength_units='Angstrom',wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3,desired_wavelength_units='Angstrom',wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit
        
    interpb1 = Interpolate(starspectrum34)
    convolveb1 = InstrumentConvolveGrating.from_grid(bosz_grid,R=24000)
    rotb1 = RotationalBroadening.from_grid(bosz_grid,vrot=np.array([10.0]))
    normb1 = Normalize(starspectrum34,2)

    interpb2 = Interpolate(starspectrum35)
    convolveb2 = InstrumentConvolveGrating.from_grid(bosz_grid,R=24000)
    normb2 = Normalize(starspectrum35,2)

    interpb3 = Interpolate(starspectrum36)
    convolveb3 = InstrumentConvolveGrating.from_grid(bosz_grid,R=24000)
    normb3 = Normalize(starspectrum36,2)


    interpp1 = Interpolate(starspectrum34)
    convolvep1 = InstrumentConvolveGrating.from_grid(phoenix_grid,R=24000)
    rotp1 = RotationalBroadening.from_grid(phoenix_grid,vrot=np.array([10.0]))
    normp1 = Normalize(starspectrum34,2)    

    interpp2 = Interpolate(starspectrum35)
    convolvep2 = InstrumentConvolveGrating.from_grid(phoenix_grid,R=24000)
    normp2 = Normalize(starspectrum35,2)

    interpp3 = Interpolate(starspectrum36)
    convolvep3 = InstrumentConvolveGrating.from_grid(phoenix_grid,R=24000)
    normp3 = Normalize(starspectrum35,2)    

    bosz_model = bosz_grid | rotb1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolveb1 & convolveb2 & convolveb3 | interpb1 & interpb2 & interpb3 | \
         normb1 & normb2 & normb3

    phoenix_model = phoenix_grid | rotp1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolvep1 & convolvep2 & convolvep3 | interpp1 & interpp2 & interpp3 | \
         normp1 & normp2 & normp3

    bosz_result = MultiNestResult.from_hdf5(boszpath)

    phoenix_result = MultiNestResult.from_hdf5(phoenixpath)

    for a in bosz_result.median.keys():
        setattr(bosz_model,a,bosz_result.median[a])

    for a in phoenix_result.median.keys():
        setattr(phoenix_model,a,phoenix_result.median[a])

    
    bw1,bf1,bw2,bf2,bw3,bf3 = bosz_model()

    pw1,pf1,pw2,pf2,pw3,pf3 = phoenix_model()
    

    bsigmas = bosz_result.calculate_sigmas(1)
    psigmas = phoenix_result.calculate_sigmas(1)


    bosz_res1 = starspectrum34.flux.value-bf1
    phoenix_res1 = starspectrum34.flux.value-pf1

    bosz_res2 = starspectrum35.flux.value-bf2
    phoenix_res2 = starspectrum35.flux.value-pf2

    bosz_res3 = starspectrum36.flux.value-bf3
    phoenix_res3 = starspectrum36.flux.value-pf3

    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))

    if starname in [x[0] for x in cal_star_info]:
        print starname+' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]


    bosz_model.teff_0 = cal_star_info[2]
    bosz_model.logg_0 = cal_star_info[3]
    bosz_model.logg_0 = cal_star_info[1]
    bosz_model.alpha_0 = cal_star_info[4]
    ap_bw1,ap_bf1,ap_bw2,ap_bf2,ap_bw3,ap_bf3 = bosz_model()


    phoenix_model.teff_0 = cal_star_info[2]
    phoenix_model.logg_0 = cal_star_info[3]
    phoenix_model.logg_0 = cal_star_info[1]
    phoenix_model.alpha_0 = cal_star_info[4]
    ap_pw1,ap_pf1,ap_pw2,ap_pf2,ap_pw3,ap_pf3 = phoenix_model()
    
    bosz_ap_res1 = starspectrum34.flux.value-ap_bf1
    phoenix_ap_res1 = starspectrum34.flux.value-ap_pf1
    
    bosz_ap_res2 = starspectrum35.flux.value-ap_bf2
    phoenix_ap_res2 = starspectrum35.flux.value-ap_pf2

    bosz_ap_res3 = starspectrum36.flux.value-ap_bf3
    phoenix_ap_res3 = starspectrum36.flux.value-ap_pf3  

    bosz_deltas = [bosz_result.median['teff_0']-cal_star_info[2], bosz_result.median['logg_0']-cal_star_info[3], bosz_result.median['mh_0']-cal_star_info[1], bosz_result.median['alpha_0']-cal_star_info[4]]
    phoenix_deltas = [phoenix_result.median['teff_0']-cal_star_info[2], phoenix_result.median['logg_0']-cal_star_info[3], phoenix_result.median['mh_0']-cal_star_info[1], phoenix_result.median['alpha_0']-cal_star_info[4]]

    plt.text(np.amax(starspectrum34.wavelength.value)-30,1.1,'PHOENIX-APOGEE param offsets:\n$\Delta T_{eff}:$'+str(phoenix_deltas[0])+'\n$\Delta log g:$'+str(phoenix_deltas[1])+\
             '\n$\Delta [M/H]:$'+str(phoenix_deltas[2])+'\n$\Delta alpha$:'+str(phoenix_deltas[3])+'\nResidual StDev:'+str(np.std(phoenix_res1)),fontsize=12)

    plt.text(np.amax(starspectrum34.wavelength.value)-30,0.1,'BOSZ-APOGEE param offsets:\n$\Delta T_{eff}:$'+str(bosz_deltas[0])+'\n$\Delta log g:$'+str(bosz_deltas[1])+'\n$\Delta [M/H]:$'+\
             str(bosz_deltas[2])+'\n$\Delta alpha$:'+str(bosz_deltas[3])+'\nResidual StDev:'+str(np.std(bosz_res1)),fontsize=12)
    
    plt.plot(starspectrum34.wavelength.value/(phoenix_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value, color='#FF0303', label = 'Data')

    plt.plot(starspectrum34.wavelength.value/(bosz_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value+0.5, color='#0394FF')

    plt.plot(starspectrum35.wavelength.value/(phoenix_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value, color='#FF0303')

    plt.plot(starspectrum35.wavelength.value/(bosz_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value+0.5, color='#0394FF')

    plt.plot(starspectrum36.wavelength.value/(phoenix_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value, color='#FF0303')

    plt.plot(starspectrum36.wavelength.value/(bosz_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value+0.5, color='#0394FF')


    plt.plot(pw1/(phoenix_result.median['vrad_3']/3e5+1.0),pf1, color='#F5A182', label = 'PHOENIX Model/Residuals')

    plt.plot(pw1/(phoenix_result.median['vrad_3']/3e5+1.0),phoenix_res1-0., color='#F5A182')

    plt.plot(pw2/(phoenix_result.median['vrad_4']/3e5+1.0),pf2, color='#F5A182')

    plt.plot(pw2/(phoenix_result.median['vrad_4']/3e5+1.0),phoenix_res2-0., color='#F5A182')

    plt.plot(pw3/(phoenix_result.median['vrad_5']/3e5+1.0),pf3, color='#F5A182')

    plt.plot(pw3/(phoenix_result.median['vrad_5']/3e5+1.0),phoenix_res3-0., color='#F5A182')
    
 
    plt.plot(bw1/(bosz_result.median['vrad_3']/3e5+1.0),bf1+0.5, color='#52D1DE', label = 'BOSZ Model/Residuals')

    plt.plot(bw1/(bosz_result.median['vrad_3']/3e5+1.0),bosz_res1+0.5, color='#52D1DE')

    plt.plot(bw2/(bosz_result.median['vrad_4']/3e5+1.0),bf2+0.5, color='#52D1DE')

    plt.plot(bw2/(bosz_result.median['vrad_4']/3e5+1.0),bosz_res2+0.5, color='#52D1DE')

    plt.plot(bw3/(bosz_result.median['vrad_5']/3e5+1.0),bf3+0.5, color='#52D1DE')
        
    plt.plot(bw3/(bosz_result.median['vrad_5']/3e5+1.0),bosz_res3+0.5, color='#52D1DE')
    

    '''
    plt.plot(ap_pw1/(phoenix_result.median['vrad_3']/3e5+1.0),phoenix_ap_res1-1.0, color='#DF1558', label = 'PHOENIX Residuals at APOGEE values')

    plt.plot(ap_pw2/(phoenix_result.median['vrad_4']/3e5+1.0),phoenix_ap_res2-1.0, color='#DF1558')

    plt.plot(ap_pw3/(phoenix_result.median['vrad_5']/3e5+1.0),phoenix_ap_res3-1.0, color='#DF1558')


    
    plt.plot(ap_bw1/(bosz_result.median['vrad_3']/3e5+1.0),bosz_ap_res1-0.5, color='#5867E5', label = 'BOSZ Residuals at APOGEE values')

    plt.plot(ap_bw2/(bosz_result.median['vrad_4']/3e5+1.0),bosz_ap_res2-0.5, color='#5867E5')

    plt.plot(ap_bw3/(bosz_result.median['vrad_5']/3e5+1.0),bosz_ap_res3-0.5, color='#5867E5')
    '''

    plt.axhline(y=0.5+0.05, color='k', linestyle='--',label='$\pm$ 5%')
    plt.axhline(y=0.5-0.05, color='k', linestyle='--')

    plt.axhline(y=-0.+0.05, color='k', linestyle='--')
    plt.axhline(y=-0.-0.05, color='k', linestyle='--')
    '''
    plt.axhline(y=-0.5+0.05, color='k', linestyle='--')
    plt.axhline(y=-0.5-0.05, color='k', linestyle='--')

    plt.axhline(y=-1.0+0.05, color='k', linestyle='--')
    plt.axhline(y=-1.0-0.05, color='k', linestyle='--')
    '''
    plt.legend(loc='upper left',fontsize=16)
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Normalized Flux')
    plt.title(starname+' PHOENIX and BOSZ fits')
    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.25,molecules=False,size=15)
    
    #bosz_result.plot_triangle(parameters=['teff_0','mh_0','logg_0','alpha_0'])
    
    #phoenix_result.plot_triangle(parameters=['teff_0','mh_0','logg_0','alpha_0'])
    print bosz_result
    print phoenix_result
    plt.show()
    


def calc_residuals(best_fit_flux, obs_flux):
    print len(best_fit_flux), len(obs_flux)
    if len(best_fit_flux) != len(obs_flux):
        best_fit_flux = signal.resample(best_fit_flux, len(obs_flux))

    residuals = obs_flux - best_fit_flux
    return residuals


def mask_data_with_residuals(flux,wavelength,uncert,residuals,limit):

    new_f = np.array([])
    new_w = np.array([])
    new_u = np.array([])
    
    for i in range(len(residuals)):
        if abs(residuals[i]) < limit:
            new_f = np.append(new_f, flux[i])
            new_w = np.append(new_w, wavelength[i])
            new_u = np.append(new_u, uncert[i])
        


    return new_f, new_w, new_u

def correct_bounds(model):
    if model.teff_0 <= model.bounds['teff_0'][0]:
        setattr(model,'teff_0',model.bounds['teff_0'][0])
    elif model.teff_0 >= model.bounds['teff_0'][1]:
        setattr(model,'teff_0',model.bounds['teff_0'][1])

    if model.logg_0 <= model.bounds['logg_0'][0]:
        setattr(model,'logg_0',model.bounds['logg_0'][0])
    elif model.logg_0 >= model.bounds['logg_0'][1]:
        setattr(model,'logg_0',model.bounds['logg_0'][1])

    if model.mh_0 <= model.bounds['mh_0'][0]:
        setattr(model,'mh_0',model.bounds['mh_0'][0])
    elif model.mh_0 >= model.bounds['mh_0'][1]:
        setattr(model,'mh_0',model.bounds['mh_0'][1])

    if model.alpha_0 <= model.bounds['alpha_0'][0]:
        setattr(model,'alpha_0',model.bounds['alpha_0'][0])
    elif model.alpha_0 >= model.bounds['alpha_0'][1]:
        setattr(model,'alpha_0',model.bounds['alpha_0'][1])

    return model




def read_nsdrp_txt(datfile,flux_units = 'erg / (cm^2 s Angstrom)', wavelength_units = 'Angstrom',
                   desired_wavelength_units = 'Angstrom',wave_range=None,clip_oh = False,clip_pix = 2,
                     clip_lines=False,single_nod=False,skip=None,normalize=True,use_snr=False):
    '''
    read the NIRSPEC dat file returned by redspec
    enter units
    if the input is a string, then open it and read, otherwise,
    iterate over the list and then average
    skip - the number of rows to skip (if different than the standard redspec output)
    '''
 
    if type(datfile) == str:

        if single_nod:
            if skip is None:
                skip =2
            # for some reason if it's a single nod like cal.dat, then there are
            # four rows at the top that needs to be skipped

            pix,wavelength,flux,snr= np.loadtxt(datfile,unpack=True,skiprows=skip,usecols=[0,1,2,5])
        else:
            if skip is None:
                skip = 2
            pix,wavelength,flux,snr= np.loadtxt(datfile,unpack=True,skiprows=skip,usecols=[0,1,2,5])
    else:
        for ii in range(len(datfile)):

            if single_nod:
                if skip is None:
                    skip = 2

                pix,wavelength,flux,snr= np.loadtxt(datfile,unpack=True,skiprows=skip,usecols=[0,1,2,5])
            else:
                if skip is None:
                    skip = 2
            pix,wavelength,flux,snr = np.loadtxt(datfile[ii],unpack=True,skiprows=skip,usecols=[0,1,2,5])
            
            if ii == 0:
                stack = np.zeros((len(flux),len(datfile)))
            if len(datfile) > 1:
                stack[:,ii] = flux
        if len(datfile) > 1:
            # take the mean along one axis
            flux = np.mean(stack,axis=1)



    if normalize:
        #p = np.polyfit(wavelength[len(flux)/20:-len(flux)/10], flux[len(flux)/20:-len(flux)/10], 1)
        norm_flux = []
        for i in range(len(flux)):
            #norm_flux += [flux[i]/(p[0]*wavelength[i]**2+p[1]*wavelength[i]**1+p[2])]
            #norm_flux += [flux[i]/(p[0]*wavelength[i]**3+p[1]*wavelength[i]**2+p[2]*wavelength[i]**1+p[3])]
            #print flux[i]/(p[0]*wavelength[i]**3+p[1]*wavelength[i]**2+p[2]*wavelength[i]**1+p[3])
            #norm_flux += [flux[i]/(p[0]*wavelength[i] + p[1])]
            norm_flux += [flux[i]/(np.median(flux))]
        flux = np.array(norm_flux)

            
    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * u.Unit(wavelength_units)
    if wave_range is not None:
        print('clipping', wave_range)
        good = np.where((wavelength.value > wave_range[0]) & (wavelength.value < wave_range[1]))[0]
        wavelength = wavelength[good]
        flux = flux[good]


    if clip_lines or clip_oh:
        # clip around OH lines
        ohlines = np.array([
            19518.4784 , 19593.2626 , 19618.5719 , 19642.4493 , 19678.046 ,
            19701.6455 , 19771.9063 , 19839.7764 ,
            20008.0235 , 20193.1799 , 20275.9409 , 20339.697 , 20412.7192 ,
            20499.237 , 20563.6072 , 20729.032 , 20860.2122 , 20909.5976 ,
            21176.5323 , 21249.5368 , 21279.1406 , 21507.1875 , 21537.4185 ,
            21580.5093 , 21711.1235 , 21802.2757 , 21873.507 , 21955.6857 ,
            22125.4484 , 22312.8204 , 22460.4183 , 22517.9267 , 22690.1765 ,
            22742.1907 , 22985.9156, 23914.55, 24041.62])
        ohclip = np.ones(len(ohlines))*clip_pix
        stellarlines = np.array([21066.9,21457.6,21898.4,22653])
        stellarclip = [4,4,4,7]  # width to clip for the lines
        if clip_oh and clip_lines:
            lines = np.append(ohlines,stellarlines)
            clip_arr = np.append(ohclip,stellarclip)
        elif clip_oh:
            lines = ohlines
            clip_arr = ohclip
        elif clip_lines:
            lines = stellarlines
            clip_pix = 6
            clip_arr = stellarclip

        lines = lines*u.angstrom
        lines = lines.to(wavelength.unit)

        deltaLambda = wavelength[1]-wavelength[0] # assume delta lambdas are uniform

        if (np.max(lines) >= np.min(lines)) & (np.min(lines) <= np.max(lines)):
            for i in np.arange(len(lines)):
                good = np.where((wavelength > lines[i]+clip_arr[i]*deltaLambda) | (wavelength < lines[i]-clip_arr[i]*deltaLambda))[0]
                if len(good) > 0:
                    flux = flux[good]
                    wavelength = wavelength[good]

    if wavelength_units != desired_wavelength_units:
        wavelength = wavelength.to(u.Unit(desired_wavelength_units))
    else:
        pass

    if use_snr:
        uncert = np.reciprocal(snr)
        spec = Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)
        spec.uncertainty = uncert*flux.unit
        return spec
    else:
        return Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)


def update_mysql_db(name, method, cutoff, grid, grid_type, orders, h5file, fit_path, passwd):
    # update the starkit database with the info from the hdf5 file and the star
    print h5file
    if os.path.exists(os.path.join(fit_path,h5file)):
        results = MultiNestResult.from_hdf5(os.path.join(fit_path,h5file))
        print results
        max = results.maximum
        med = results.median
        sig = results.calculate_sigmas(1)
        if 'add_err_15' in max.keys():
            p = ['teff_0','logg_0','mh_0','alpha_0','vrot_1','add_err_15']
        else:
            p = ['teff_0','logg_0','mh_0','alpha_0','vrot_1']

        if '2MJ18113-30441' in name:
            nirspec_stat = True
        else:
            nirspec_stat = False
        chi2, max_res, mean_res = calc_chi2_for_sql(name,results,grid,nirspec_upgrade=nirspec_stat)
            
        temp = []
        for k in p:
            temp.extend([float(max[k]),float(med[k]),(sig[k][1]-sig[k][0])/2.0,sig[k][1],sig[k][0]])

        values = [name,'k-band',method,cutoff,grid_type,orders] + temp

        values += [mean_res,max_res,chi2]

        print values
#        if 'add_err_6' in m.keys():
#            values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
#                     temp[7] + [original_location,spectrum_file,h5file,str(datetime.datetime.today())]
#        else:
#            values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
#                     [None,None,None,None] + [original_location,spectrum_file,h5file,str(datetime.datetime.today())]

        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbwrite',passwd=passwd,db='gcg')
        cur = con.cursor()

        #update_values = [name,'k-band',int(cutoff),grid,orders,round(med['teff_0'],2),round(m['teff_0'],2)]

        for i in values:
            print type(i)


        if 'add_err_15' in max.keys():
            sql_query = 'INSERT INTO metallicity (name,band,masking_method,cutoff,grid,orders,'+\
                    'teff_peak,teff,teff_err,teff_err_upper,teff_err_lower,'+\
                    'logg_peak,logg,logg_err,logg_err_upper,logg_err_lower,'+\
                    'mh_peak,mh,mh_err,mh_err_upper,mh_err_lower,'+\
                    'alpha_peak,alpha,alpha_err,alpha_err_upper,alpha_err_lower,'+\
                    'vrot_peak,vrot,vrot_err,vrot_err_upper,vrot_err_lower,'+ \
                    'adderr_peak,adderr,adderr_err,adderr_err_upper,adderr_err_lower,' + \
                    'residual_mean,residual_max,reduced_chi2) '+\
                    'VALUES(%s,%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+ \
                    '%s,%s,%s,%s,%s,' + \
                    '%s,%s,%s,%s,%s,' + \
                    '%s,%s,%s)'
        else:
            sql_query = 'INSERT INTO metallicity (name,band,masking_method,cutoff,grid,orders,'+\
                    'teff_peak,teff,teff_err,teff_err_upper,teff_err_lower,'+\
                    'logg_peak,logg,logg_err,logg_err_upper,logg_err_lower,'+\
                    'mh_peak,mh,mh_err,mh_err_upper,mh_err_lower,'+\
                    'alpha_peak,alpha,alpha_err,alpha_err_upper,alpha_err_lower,'+\
                    'vrot_peak,vrot,vrot_err,vrot_err_upper,vrot_err_lower,'+ \
                    'residual_mean,residual_max,reduced_chi2) '+\
                    'VALUES(%s,%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,'+ \
                    '%s,%s,%s,%s,%s,' + \
                    '%s,%s,%s)'


        print('adding into db',type(values))
        print(len(values))
        print sql_query
        print sql_query % tuple(values)

        #print testval
        cur.execute(sql_query,values)

        con.commit()
        cur.close()
        con.close()

def add_column_mysql_db(h5file, fit_path, passwd, id):
    # update the starkit database with the info from the hdf5 file and the star
    print h5file
    if os.path.exists(os.path.join(fit_path, h5file)):
        results = MultiNestResult.from_hdf5(os.path.join(fit_path, h5file))
        print (h5file)
        print results
        max = results.maximum
        med = results.median
        sig = results.calculate_sigmas(1)
        if 'add_err_15' in max.keys():
            values = [h5file,float(max['add_err_15']), float(med['add_err_15']), (sig['add_err_15'][1] - sig['add_err_15'][0]) / 2.0, sig['add_err_15'][1], sig['add_err_15'][0], id]

        else:
            values = [h5file, id]


        con = mdb.connect(host='galaxy1.astro.ucla.edu', user='dbwrite', passwd=passwd, db='gcg')
        cur = con.cursor()

        # update_values = [name,'k-band',int(cutoff),grid,orders,round(med['teff_0'],2),round(m['teff_0'],2)]

        '''
        if 'add_err_15' in max.keys():
            sql_query = 'UPDATE metallicity SET filename = %s WHERE teff_peak = %s,teff = %s,teff_err = %s,teff_err_upper = %s,teff_err_lower = %s,' + \
                        'logg_peak = %s,logg = %s,logg_err = %s,logg_err_upper = %s,logg_err_lower = %s,' + \
                        'mh_peak = %s,mh = %s,mh_err = %s,mh_err_upper = %s,mh_err_lower = %s,' + \
                        'alpha_peak = %s,alpha = %s,alpha_err = %s,alpha_err_upper = %s,alpha_err_lower = %s,' + \
                        'vrot_peak = %s,vrot = %s,vrot_err = %s,vrot_err_upper = %s,vrot_err_lower = %s,' + \
                        'adderr_peak = %s,adderr = %s,adderr_err = %s,adderr_err_upper = %s,adderr_err_lower = %s'

        else:
            sql_query = 'UPDATE metallicity SET filename = %s WHERE teff_peak = %s,teff = %s,teff_err = %s,teff_err_upper = %s,teff_err_lower = %s,' + \
                        'logg_peak = %s,logg = %s,logg_err = %s,logg_err_upper = %s,logg_err_lower = %s,' + \
                        'mh_peak = %s,mh = %s,mh_err = %s,mh_err_upper = %s,mh_err_lower = %s,' + \
                        'alpha_peak = %s,alpha = %s,alpha_err = %s,alpha_err_upper = %s,alpha_err_lower = %s,' + \
                        'vrot_peak = %s,vrot = %s,vrot_err = %s,vrot_err_upper = %s,vrot_err_lower = %s'
        '''

        if 'add_err_15' in max.keys():
            sql_query = 'UPDATE metallicity SET filename = %s,adderr_peak = %s,adderr = %s,adderr_err = %s,adderr_err_upper = %s,adderr_err_lower = %s WHERE id = %s'
        else:
            sql_query = 'UPDATE metallicity SET filename = %s WHERE id = %s'

        print('adding into db', type(values))
        print sql_query
        print sql_query % tuple(values)

        # print testval
        cur.execute(sql_query, values)

        con.commit()
        cur.close()
        con.close()

def calc_chi2_for_sql(starname,result, grid, nirspec_upgrade=False):
    specdir = '/u/rbentley/metallicity/spectra/'
    snr = 30.
    if not nirspec_upgrade:
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)


        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
                snr)) * starspectrum36.flux.unit

    else:
        print specdir + starname + '_order34*.dat'
        file1 = glob.glob(specdir + starname + '_order34*.dat')
        file2 = glob.glob(specdir + starname + '_order35*.dat')
        file3 = glob.glob(specdir + starname + '_order36*.dat')

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom')
        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom')
        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom')

        waverange34 = [np.amin(starspectrum34.wavelength.value[20:2000]),
                       np.amax(starspectrum34.wavelength.value[:2000])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[20:2000]),
                       np.amax(starspectrum35.wavelength.value[:2000])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[20:2000]),
                       np.amax(starspectrum36.wavelength.value[:2000])]

        starspectrum34 = read_nsdrp_txt(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit

        starspectrum35 = read_nsdrp_txt(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit

        starspectrum36 = read_nsdrp_txt(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

    interp1 = Interpolate(starspectrum34)
    convolve1 = InstrumentConvolveGrating.from_grid(grid, R=24000)
    rot1 = RotationalBroadening.from_grid(grid, vrot=np.array([10.0]))
    norm1 = Normalize(starspectrum34, 2)

    interp2 = Interpolate(starspectrum35)
    convolve2 = InstrumentConvolveGrating.from_grid(grid, R=24000)
    norm2 = Normalize(starspectrum35, 2)

    interp3 = Interpolate(starspectrum36)
    convolve3 = InstrumentConvolveGrating.from_grid(grid, R=24000)
    norm3 = Normalize(starspectrum36, 2)


    model = grid | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
                 convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
                 norm1 & norm2 & norm3

    for a in result.median.keys():
        setattr(model, a, result.median[a])

    w1, f1, w2, f2, w3, f3 = model()

    res1 = starspectrum34.flux.value - f1

    res2 = starspectrum35.flux.value - f2

    res3 = starspectrum36.flux.value - f3

    chi2 = np.sum((res1)**2/(starspectrum34.uncertainty.value)**2)
    chi2 = chi2 + np.sum((res2)**2/(starspectrum35.uncertainty.value)**2)
    chi2 = chi2 + np.sum((res3)**2/(starspectrum36.uncertainty.value)**2)
    chi2  = chi2/(len(res1)+len(res2)+len(res3))

    total_res = np.concatenate((res1,res2,res3))

    total_res_abs = np.absolute(total_res)

    max_res = np.amax(total_res)

    mean_res = np.mean(total_res_abs)

    return chi2, max_res, mean_res

'''




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
    elif param is 'alpha':
        model.mh_0 = param_val
    

    R1_cen,R2_cen,R3_cen,R4_cen = r_val_polynomial_four_order(model)

    if param is 'teff':
        model.teff_0 = param_val+increment
    elif param is 'logg':
        model.logg_0 = param_val+increment
    elif param is 'mh':
        model.mh_0 = param_val+increment
    elif param is 'alpha':
        model.mh_0 = param_val_increment

    R1_up, R2_up, R3_up,R4_up = r_val_polynomial_four_order(model)

    if param is 'teff':
        model.teff_0 = param_val-increment
    elif param is 'logg':
        model.logg_0 = param_val-increment
    elif param is 'mh':
        model.mh_0 = param_val-increment
    elif param is 'alpha':
        model.mh_0 = param_val-increment

    R1_dw,R2_dw,R3_dw,R4_dw = r_val_polynomial_four_order(model)
    
    s_lambda1 = [100*(R1_up[i] - R1_dw[i])/R1_cen[i] for i in range(len(R1_up))]

    s_lambda2 = [100*(R2_up[i] - R2_dw[i])/R2_cen[i] for i in range(len(R2_up))]

    s_lambda3 = [100*(R3_up[i] - R3_dw[i])/R3_cen[i] for i in range(len(R3_up))]
    
    s_lambda4 = [100*(R4_up[i] - R4_dw[i])/R4_cen[i] for i in range(len(R4_up))]

    return s_lambda1, s_lambda2, s_lambda3, s_lambda4
'''
