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
#sns.set_context('paper',font_scale=2.0, rc={"lines.linewidth": 1.75})
#sns.set_style("white")
#sns.set_style('ticks')

directory = '/group/data/nirspec/'
dbfile = os.path.join(directory,'nirspec_fit_params.sqlite')

def get_grid(gridfile='/u/tdo/research/phoenix/phoenix_r40000_1.9-2.5_k.h5'):
    # use this so that we only need to load the grid once.
    # in ipython call
    # g = fit_nirspec.get_grid()
    #
    # then, you can use the functions as usual, but passing in g
    # fit_nirspec.fit_all_star(g=g)

    # for fit in paper
    # g=fit_nirspec.get_grid(gridfile='../../phoenix/phoenix_r40000_alpha_logg4_k.h5')
    return load_grid(gridfile)

def fit_all_stars(g=None,directory='/group/data/nirspec/spectra/',order=35,snr=35,savedir='../nirspec_fits/',l1norm=False):
    #g=load_grid('/u/tdo/research/phoenix/phoenix_r100000_order34.h5')   # has alpha = 0
    if g is None:
        g=load_grid('/u/tdo/research/phoenix/phoenix_r40000_1.9-2.5_k.h5')
    #names = ['NE_1_002']
    #names = ['NE_1_002']
    #names = ['NGC6819_J19213390+3750202']
    #names = ['M71_J19534827+1848021']
    names = ['NGC6791_J19205+3748282']
    wave_ranges = {'order35':[2.178,2.20882],'order34':[2.2406, 2.271]}
    wave_range = wave_ranges['order'+str(order)]  # order 35

    plt.clf()
    for i in xrange(len(names)):
        filename = glob.glob(directory+names[i]+'_order'+str(order)+'*.dat')
        starspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
        starspectrum.uncertainty = (np.zeros(len(starspectrum.flux.value))+1.0/np.float(snr))*starspectrum.flux.unit
        print starspectrum.wavelength.unit
        print starspectrum.wavelength[0:10]
        plt.plot(starspectrum.wavelength,starspectrum.flux)
        result = fit_star(starspectrum,g,R=30000.0,logg_range=[1.5,4.4],vrad_range=[-1000,1000],l1norm=l1norm)
        print result.median
        print result.calculate_sigmas(1)
        outfile = os.path.join(savedir,filename[0]+'.h5')
        print 'saving file: '+ outfile
        result.to_hdf(outfile)

def fit_quinn_star(g=None,mkplot=False):
    # fit Quinn's starspectrum
    if g is None:
        # NOTE: load the grid in ipython in order to make it go faster and save memory
        g=load_grid('/u/tdo/research/phoenix/phoenix_r40000_1.9-2.5_k.h5')
    filename = '../quinn/quinn_nirspao.txt'
    run_dir = '../quinn/chains/'
    wave_range=[2.292082,2.292280]
    flux_units = 'erg / (cm^2 s Angstrom)'
    wavelength_units = 'Angstrom'
    snr = 60.0
    wavelength,flux = np.loadtxt(filename,unpack=True)
    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * 1e4* u.Unit(wavelength_units)
    starspectrum = Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)
    starspectrum.uncertainty = (np.zeros(len(starspectrum.flux.value))+1.0/np.float(snr))*starspectrum.flux.unit


    my_model = assemble_model(g, vrad=200, vrot=0,R=20000,spectrum=starspectrum,normalize_npol=1)
    my_model.teff_0 = 3377.0
    my_model.mh_0 = 0.16
    my_model.logg_0 = 3.46
    my_model.vrot_1 = 1.0
    my_model.R_3 = 20000.0
    my_model.vrad_2 = -1.464
    wave,flux = my_model()
    if mkplot:
        plt.clf()
        plt.plot(starspectrum.wavelength,starspectrum.flux)
        plt.plot(wave,flux)
        plt.xlabel('Wavelength (Angstrom)')
        plt.ylabel('Flux')

    # uncomment below to fit
    #fit_star(starspectrum,g,run_dir=run_dir,logg_range=[3.0,4.0],R=20000.0)


    # this is the results
    # OrderedDict([(u'teff_0', (3377.041312970417, 3377.0672616175243)), (u'logg_0', (3.457898218197375, 3.4579304308093466)), (u'mh_0', (0.15676195245458976, 0.15686878469571527)), (u'vrot_1', (0.2821351844695545, 1.5631034890489999)), (u'vrad_2', (-1.4640572701007544, -1.4637592900842216)), (u'R_3', (20000.0, 20000.0))])
def model_example():
    # plot some example spctra from the model
    g=load_grid('/u/tdo/research/phoenix/phoenix_r40000_1.9-2.5_k.h5')   # has alpha = 0
    names = ['NE_1_002']
    wave_range = [2.178,2.20882]
    filename = directory+names[0]+'.tar.dat'
    starspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')

    my_model = assemble_model(g, vrad=0, vrot=0,R=20000,spectrum=starspectrum,normalize_npol=1)
    my_model.teff_0 = 3500.0
    my_model.mh_0 = 0.0
    my_model.logg_0 = 2.0
    my_model.vrot_1 = 0.0
    my_model.R_3 = 40000.0
    my_model.vrad_2 = -240.0
    wave,flux = my_model()


    plt.clf()
    plt.plot(wave,flux,label='[M/H] = 0.0')
    my_model.mh_0 = 0.4
    wave,flux = my_model()
    plt.plot(wave,flux,label='[M/H] = 0.4')

    my_model.mh_0 = -1.0
    wave,flux = my_model()
    plt.plot(wave,flux,label='[M/H] = -1.0')

    plt.plot(starspectrum.wavelength,starspectrum.flux,label='Observed')
    plt.legend(loc=3)


def fit_star(starspectrum,grid,teff_range=[3000,6000],vrad_range=[-200,200],
             logg_range=[1.5,4.4],mh_range=[-1.5,1.0],vrot_range=[0,20],
             R=4000,run_dir=None,resume=False,verbose=True,l1norm=False,
             normalize_npol=2,alpha_range=[-0.4,1.2],savefile=False):
    # fit a star

    my_model = assemble_model(grid, vrad=0, vrot=0,R=R,spectrum=starspectrum,normalize_npol=normalize_npol)
    wave,flux = my_model()
    #plt.plot(wave,flux)

    teff_prior = priors.UniformPrior(teff_range[0],teff_range[1])
    logg_prior = priors.UniformPrior(logg_range[0],logg_range[1])

    mh_prior = priors.UniformPrior(mh_range[0],mh_range[1])
    #mh_prior = priors.FixedPrior(0.8)
    alpha_prior = priors.UniformPrior(*alpha_range)

    vrad_prior = priors.UniformPrior(vrad_range[0],vrad_range[1])
    vrot_prior = priors.UniformPrior(vrot_range[0],vrot_range[1])
    R_prior = priors.FixedPrior(R)

    if 'alpha' in grid.param_names:
        prior_list = [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, vrad_prior,R_prior]
    else:
        prior_list = [teff_prior, logg_prior, mh_prior, vrot_prior, vrad_prior,R_prior]
    # assemble likelihood and the model
    if l1norm:
        ll = L1Likelihood(starspectrum)
    else:
        ll = Chi2Likelihood(starspectrum)

    fit_model = my_model | ll

    #fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior,vrot_prior, vrad_prior,  R_prior])
    fitobj = MultiNest(fit_model, prior_list,run_dir=run_dir)
    result = fitobj.run(verbose=verbose,clean_up=False,resume=resume)

    mean_vals = result.median
    fit_model.teff_0 = mean_vals['teff_0']
    fit_model.logg_0 = mean_vals['logg_0']
    if 'alpha' in grid.param_names:
        fit_model.alpha_0 = mean_vals['alpha_0']
    fit_model.vrot_1 = mean_vals['vrot_1']
    fit_model.vrad_2 = mean_vals['vrad_2']
    fit_model.R_3 = mean_vals['R_3']

    fit_wave,fit_flux = fit_model[:-1]()
    plt.plot(fit_wave,fit_flux)
    #print result.calculate_sigmas(1)

    # the posteriors are stored as a pandas data frame
    posterior = result.posterior_data
    #weights = posterior['posterior']  # chain weights
    teff = posterior['teff_0']

    # to make the triangle plots
    #f.plot_triangle(extents=[0.999,0.999,0.999,0.999,0.999,0.999],plot_contours=False)

    # can also save the changes into an hdf5 file:
    # result.to_hdf('results.h5')
    return result

class Splitter2(Model):
    # split a single spectrum into 2
    inputs=('w', 'f')
    outputs = ('w', 'f', 'w', 'f')
    def evaluate(self, w, f):
        return w,f,w,f

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

def fit_star_multi_order(starname,g,specdir='/group/data/nirspec/spectra/',
    savedir='../nirspec_fits/',snr=30.0,nnorm=2,teff_range=[3000,6000],
    vrad_range=[-600,600],logg_range=[1.5,4.0],mh_range=[-1.5,1.0],vrot_range=[0,20],
    R=30000,verbose=True,alpha_range=[-0.2,1.2],r_range=[15000.0,40000.0],
                         R_fixed=None,logg_fixed=None,l1norm=False):

    # fit a spectrum of a star with multiple orders that can have different velocities
    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')
    file4 = glob.glob(specdir+starname+'_order37*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=[2.2406, 2.271])
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=[2.178,2.20882])
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=[2.11695,2.14566])
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='Angstrom',wave_range=[2.0750,2.0875])
    starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit



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


    plt.clf()
    plt.plot(w1,f1)
    plt.plot(starspectrum34.wavelength,starspectrum34.flux)
    plt.plot(w2,f2)
    plt.plot(starspectrum35.wavelength,starspectrum35.flux)

    plt.plot(w3,f3)
    plt.plot(starspectrum36.wavelength,starspectrum36.flux)

    plt.plot(w4,f4)
    plt.plot(starspectrum37.wavelength,starspectrum37.flux)

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

    fitobj.run()
    result=fitobj.result

    if l1norm:
        like_str = '_l1norm'
    else:
        like_str = ''
    result.to_hdf(os.path.join(savedir,starname+'_results'+like_str+'.h5'))
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

    file1 = os.path.join(savedir,starname+'_order34_model.txt')
    file2 = os.path.join(savedir,starname+'_order35_model.txt')
    file3 = os.path.join(savedir,starname+'_order36_model.txt')
    file4 = os.path.join(savedir,starname+'_order37_model.txt')

    write_spectrum.write_txt(w1,f1,file1,comments=comment1)
    write_spectrum.write_txt(w2,f2,file2,comments=comment2)
    write_spectrum.write_txt(w3,f3,file3,comments=comment3)
    write_spectrum.write_txt(w4,f4,file4,comments=comment4)


def batch_fit_star_multi(g):
    starnames = ['NE1-1 001','NE1-1 002', 'NGC6791 J19213390+3750202']
    stars = ['NE_1_001','NE_1_002','NGC6791_J19205+3748282']
    #starnames = ['NE1-1 001']
    #stars = ['NE_1_001']

    R_fixed = [25000.0, 25000.0,25000.0]
    #logg_fixed = [1.0,1.0,1.3]
    logg_fixed = None
    logg_range = [1.0, 1.5]
    teff_range = [3300.0, 7000.0]
    for i in np.arange(len(starnames)):
        s = stars[i]
        print("fitting star: "+s)
        if logg_fixed is not None:
            lf = logg_fixed[i]
        else:
            lf = None
            
        fit_star_multi_order(s,g,logg_range=logg_range,logg_fixed=lf,R_fixed=R_fixed[i],
                             teff_range=teff_range)

def plot_multi_order_fit(starname,g=None,savefile=None,specdir='/group/data/nirspec/spectra/',
    savedir = '../nirspec_fits/',snr=30.0,nnorm=2,save_model=False):
    # plot the results of a multiple order fit on observed spectrum.

    file1 = glob.glob(specdir+starname+'_order34*.dat')
    file2 = glob.glob(specdir+starname+'_order35*.dat')
    file3 = glob.glob(specdir+starname+'_order36*.dat')
    file4 = glob.glob(specdir+starname+'_order37*.dat')

    if savefile is None:
        savefile = os.path.join(savedir,starname+'_results.h5')
    # restore MultiNest savefile
    result = MultiNestResult.from_hdf5(savefile)

    starspectrum34 = read_fits_file.read_nirspec_dat(file1,desired_wavelength_units='Angstrom',wave_range=[2.2406, 2.271])
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value))+1.0/np.float(snr))*starspectrum34.flux.unit
    starspectrum35 = read_fits_file.read_nirspec_dat(file2,desired_wavelength_units='Angstrom',wave_range=[2.178,2.20882])
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value))+1.0/np.float(snr))*starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3,desired_wavelength_units='Angstrom',wave_range=[2.11695,2.14566])
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    starspectrum37 = read_fits_file.read_nirspec_dat(file4,desired_wavelength_units='Angstrom',wave_range=[2.0750,2.0875])
    starspectrum37.uncertainty = (np.zeros(len(starspectrum37.flux.value))+1.0/np.float(snr))*starspectrum36.flux.unit

    if g is not None:
        interp1 = Interpolate(starspectrum34)
        convolve1 = InstrumentConvolveGrating.from_grid(g,R=30000)
        rot1 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm1 = Normalize(starspectrum34,nnorm)

        interp2 = Interpolate(starspectrum35)
        convolve2 = InstrumentConvolveGrating.from_grid(g,R=30000)
        #rot2 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm2 = Normalize(starspectrum35,nnorm)

        interp3 = Interpolate(starspectrum36)
        convolve3 = InstrumentConvolveGrating.from_grid(g,R=30000)
        #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm3 = Normalize(starspectrum36,nnorm)

        interp4 = Interpolate(starspectrum37)
        convolve4 = InstrumentConvolveGrating.from_grid(g,R=30000)
        #rot4 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
        norm4 = Normalize(starspectrum37,nnorm)



        model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
             convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
             norm1 & norm2 & norm3 & norm4
        model.teff_0 = result.median['teff_0']
        model.logg_0 = result.median['logg_0']
        model.mh_0 = result.median['mh_0']
        model.alpha_0 = result.median['alpha_0']
        model.vrot_1 = result.median['vrot_1']
        model.vrad_3 = result.median['vrad_3']
        model.vrad_4 = result.median['vrad_4']
        model.vrad_5 = result.median['vrad_5']
        model.vrad_6 = result.median['vrad_6']
        model.R_7 = result.median['R_7']
        model.R_8 = result.median['R_8']
        model.R_9 = result.median['R_9']
        model.R_10 = result.median['R_10']

        w1,f1,w2,f2,w3,f3,w4,f4 = model()

    else:

        file1 = os.path.join(savedir,starname+'_order34_model.txt')
        file2 = os.path.join(savedir,starname+'_order35_model.txt')
        file3 = os.path.join(savedir,starname+'_order36_model.txt')
        file4 = os.path.join(savedir,starname+'_order37_model.txt')

        w1,f1 = np.loadtxt(file1,usecols=(0,1),unpack=True)
        w2,f2 = np.loadtxt(file2,usecols=(0,1),unpack=True)
        w3,f3 = np.loadtxt(file3,usecols=(0,1),unpack=True)
        w4,f4 = np.loadtxt(file4,usecols=(0,1),unpack=True)


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
    observed_wave = (starspectrum37.wavelength,starspectrum36.wavelength,
                     starspectrum35.wavelength,starspectrum34.wavelength)
    observed_flux = (starspectrum37.flux,starspectrum36.flux,
                     starspectrum35.flux,starspectrum34.flux)
    model_wave = (w4,w3,w2,w1)
    model_flux = (f4,f3,f2,f1)
    max_result = result.maximum
    vels = (max_result['vrad_6'],max_result['vrad_5'],max_result['vrad_4'],max_result['vrad_3'])

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

    if save_model:
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

        file1 = os.path.join(savedir,starname+'_order34_model.txt')
        file2 = os.path.join(savedir,starname+'_order35_model.txt')
        file3 = os.path.join(savedir,starname+'_order36_model.txt')
        file4 = os.path.join(savedir,starname+'_order37_model.txt')

        write_spectrum.write_txt(w1,f1,file1,comments=comment1)
        write_spectrum.write_txt(w2,f2,file2,comments=comment2)
        write_spectrum.write_txt(w3,f3,file3,comments=comment3)
        write_spectrum.write_txt(w4,f4,file4,comments=comment4)

def fit_star_prev(starspectrum,grid):
    # fit a star

    my_model = assemble_model(grid, vrad=0, vrot=0,R=20000,spectrum=starspectrum,normalize_npol=2)
    wave,flux = my_model()
    #plt.plot(wave,flux)

    teff_prior = priors.UniformPrior(3000,6000)
    logg_prior = priors.UniformPrior(0.5,4.0)

    mh_prior = priors.UniformPrior(-1.5,1.0)
    #mh_prior = priors.FixedPrior(0.8)
    #alpha_prior = priors.UniformPrior(-0.2,1.2)

    vrad_prior = priors.UniformPrior(-600.0,600.0)
    vrot_prior = priors.UniformPrior(0,20)
    #R_prior = priors.FixedPrior(80000)
    R_prior = priors.UniformPrior(20000,100000)

    # assemble likelihood and the model
    ll = Chi2Likelihood(starspectrum)
    fit_model = my_model | ll

    #fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior,vrot_prior, vrad_prior,  R_prior])
    fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, vrot_prior, vrad_prior,R_prior])
    result = fitobj.run(verbose=True,clean_up=True)

    mean_vals = result.mean
    fit_model.teff_0 = mean_vals['teff_0']
    fit_model.logg_0 = mean_vals['logg_0']
    fit_model.vrot_1 = mean_vals['vrot_1']
    fit_model.vrad_2 = mean_vals['vrad_2']
    fit_model.R_3 = mean_vals['R_3']

    fit_wave,fit_flux = fit_model[:-1]()
    plt.plot(fit_wave,fit_flux)
    print result.calculate_sigmas(1)

    # the posteriors are stored as a pandas data frame
    posterior = result.posterior_data
    #weights = posterior['posterior']  # chain weights
    teff = posterior['teff_0']

    # to make the triangle plots
    #f.plot_triangle(extents=[0.999,0.999,0.999,0.999,0.999,0.999],plot_contours=False)

    # can also save the changes into an hdf5 file:
    # result.to_hdf('results.h5')
    return result

def plot_gc_cluster_compare():
    # plot spectra from GC stars compared to open clusters

    #directory='/u/tdo/data/gc/nirspec/spectra/'
    specdir = os.path.join(directory,'spectra')
    gc = 'NE_1_002'
    gc_label = 'NE_1_002 [M/H]'
    gc2 = 'NE_1_003'
    gc2_label = 'NE_1_003 [M/H] < -1.0'

    compare = 'NGC6791_J19213390+3750202'
    compare_label = 'NGC6791 J19213390+3750202 [M/H] = 0.33'


    order = 34

    if order == 34:
        v1 = -540.0
        v1a = -540.0
        v2 = -458.0
        wave_range = [2.2406, 2.271]
        xlim = np.array([2.2445,2.271])*1e4
    if order == 35:
        v1 = -240.0
        v1a = -221.0
        v2 = 141.0
        wave_range = [2.178,2.20882]
        xlim = np.array([[2.1804,2.1960],[2.1992,2.2103]])*1e4

    filename = glob.glob(specdir+'/'+gc+'_order'+str(order)+'*.dat')
    gcspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')

    gcspectrum.wavelength = gcspectrum.wavelength*(-v1/3e5+1.0)

    filename2 = glob.glob(specdir+'/'+gc2+'_order'+str(order)+'*.dat')

    gcspectrum2 = read_fits_file.read_nirspec_dat(filename2,wave_range=wave_range,desired_wavelength_units='Angstrom')

    gcspectrum2.wavelength = gcspectrum2.wavelength*(-v1a/3e5+1.0)

    filename = glob.glob(specdir+'/'+compare+'_order'+str(order)+'*.dat')

    comparespectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
    comparespectrum.wavelength = comparespectrum.wavelength*(-v2/3e5+1.0)
    plt.clf()
    if len(np.shape(xlim)) == 2:
        plt.subplot(2,1,1)
    plt.plot(gcspectrum.wavelength,gcspectrum.flux,label=gc_label)
    #plt.plot(gcspectrum2.wavelength,gcspectrum2.flux,label=gc2_label)
    plt.plot(comparespectrum.wavelength,comparespectrum.flux,label=compare_label)
    plt.ylim(0.2,1.4)
    if len(np.shape(xlim)) == 2:
        plt.xlim(xlim[0,0],xlim[0,1])
    else:
        plt.legend(loc=2)
        plt.xlim(xlim[0],xlim[1])

    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5)

    plt.ylabel('Flux')
    plt.xlabel('Wavelength (Angstrom)')

    if len(np.shape(xlim)) == 2:
        plt.subplot(2,1,2)
        plt.plot(gcspectrum.wavelength,gcspectrum.flux,label=gc_label)
        plt.plot(gcspectrum2.wavelength,gcspectrum2.flux,label=gc2_label)
        plt.plot(comparespectrum.wavelength,comparespectrum.flux,label=compare_label)
        plt.xlim(xlim[1,0],xlim[1,1])
        plt.ylim(0.2,1.4)
        plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5)
        plt.legend(loc=2)
        plt.ylabel('Flux')
        plt.xlabel('Wavelength (Angstrom)')

    plt.tight_layout()

def plot_gc_cluster_low_mh():
    # plot spectra from GC stars compared to open clusters

    #directory='/u/tdo/data/gc/nirspec/spectra/'
    specdir = os.path.join(directory,'spectra')
    gc = 'NE_1_002'
    gc_label = 'NE1-002'   # high metallicity
    gc2 = 'NE_1_003'
    gc2_label = 'NE1-003'  # low metallicity

    compare = 'M71_J19534827+1848021'
    compare_label = 'M71 J19534827+1848021 [M/H] = -0.7'

    compare2 = 'NGC6791_J19213390+3750202'
    compare_label2 = 'NGC6791 J19213390+3750202 [M/H] = 0.33'

    order = 34

    if order == 34:
        v1 = -540.0
        v1a = -520.0
        v2 = -438.5
        v2a = -458.0
        wave_range = [2.24, 2.275]
        xlim =  np.array([2.2474, 2.2678])*1e4
        #xlim = np.array(wave_range)*1e4
    if order == 35:
        v1 = -240.0
        v1a = -221.0
        v2 = 161.0
        v2a = 141.0
        wave_range = [2.178,2.20882]
        xlim = np.array([2.1880,2.2076])*1e4
        #xlim = np.array([[2.1804,2.1960],[2.1992,2.2103]])*1e4

    filename = glob.glob(specdir+'/'+gc+'_order'+str(order)+'*.dat')
    gcspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')

    gcspectrum.wavelength = gcspectrum.wavelength*(-v1/3e5+1.0)

    filename2 = glob.glob(specdir+'/'+gc2+'_order'+str(order)+'*.dat')

    gcspectrum2 = read_fits_file.read_nirspec_dat(filename2,wave_range=wave_range,desired_wavelength_units='Angstrom')

    gcspectrum2.wavelength = gcspectrum2.wavelength*(-v1a/3e5+1.0)

    filename = glob.glob(specdir+'/'+compare+'_order'+str(order)+'*.dat')

    comparespectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
    comparespectrum.wavelength = comparespectrum.wavelength*(-v2/3e5+1.0)

    filename = glob.glob(specdir+'/'+compare2+'_order'+str(order)+'*.dat')
    comparespectrum2 = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
    comparespectrum2.wavelength = comparespectrum2.wavelength*(-v2a/3e5+1.0)

    plt.clf()
    
    plt.subplot(2,1,1)
    plt.ylim(0.2,1.4)



    #plt.plot(gcspectrum.wavelength,gcspectrum.flux,label=gc_label)
    plt.plot(comparespectrum.wavelength,comparespectrum.flux,label=compare_label)

    plt.plot(gcspectrum2.wavelength,gcspectrum2.flux,label=gc2_label)

    plt.xlim(xlim[0],xlim[1])
    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5)

    plt.legend(loc=2)
    plt.ylabel('Flux')
    plt.xlabel('Wavelength (Angstrom)')


    plt.subplot(2,1,2)
    plt.plot(comparespectrum2.wavelength,comparespectrum2.flux,label=compare_label2)
    plt.plot(gcspectrum.wavelength,gcspectrum.flux,label=gc_label)
    #plt.plot(gcspectrum2.wavelength,gcspectrum2.flux,label=gc2_label)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(0.2,1.4)
    plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5)

    plt.legend(loc=2)
    plt.ylabel('Flux')
    plt.xlabel('Wavelength (Angstrom)')

    plt.tight_layout()

def plot_ngc6791_grid_compare():
    # plot spectra from GC stars compared to open clusters

    #directory='/u/tdo/data/gc/nirspec/spectra/'
    specdir = os.path.join(directory,'spectra')
    compare2 = 'NGC6791_J19213390+3750202'
    compare_label2 = 'NGC6791 J19213390+3750202 [M/H] = 0.33, [alpha/Fe] = 0.22'

    ## infiles = ['sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_-0.50.fits',
    ##            'sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_0.00.fits',
    ##            'sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_0.20.fits',
    ##            'sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_0.50.fits',
    ##            'sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_1.00.fits']
    ## labels = ['-0.50','0.00','0.20','0.50','1.00']
    infiles = ['sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_0.00.fits',
               'sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_0.50.fits',
               'sme_grid_ngc6791_teff_3901._logg_1.26_feh_0.33_alpha_0.22_sc_1.00.fits']
    labels = ['0.00','0.50','1.00']    
    indir = '/u/tdo/research/sme_grid/grid/'
    order = 35
    if order == 34:
        v1 = -540.0
        v1a = -540.0
        v2 = -438.5
        v2a = -458.0
        wave_range = [2.2406, 2.271]
        xlim = np.array(wave_range)*1e4
    if order == 35:
        v1 = -240.0
        v1a = -221.0
        v2 = 161.0
        v2a = 141.0
        wave_range = [2.178,2.20882]
        xlim = np.array([2.1880,2.2076])*1e4
        xlim = [2.1993, 2.20849] # this shows the zoomed in on Sc

    filename = glob.glob(specdir+'/'+compare2+'_order'+str(order)+'*.dat')
    comparespectrum2 = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
    wave = comparespectrum2.wavelength.value/1e4
    flux = comparespectrum2.flux.value
    print comparespectrum2.flux[0:10]
    #plot_range = [2.201,2.209]
    plot_range = [2.180,2.207]
    plot_range = wave_range
    plt.xlim(xlim[0],xlim[1])
    plot_abundance_grid(wave,flux,plot_range=xlim,vel=v2a,starname=compare_label2,
                        infiles=infiles,indir=indir,labels=labels,R=2500.0,size=12)

    ticks = np.linspace(xlim[0],xlim[1],5) # five tick marks
    tick_labels = ['%8.4f' % (n) for n in ticks]
    plt.xticks(ticks,tick_labels)
    #plot_abundance_grid(comparespectrum2.wavelength.value,comparespectrum2.flux.value,vel=v2a)

def plot_gc_grid_compare(order=34):
    # plot spectra from GC stars compared to the SME grid

    #directory='/u/tdo/data/gc/nirspec/spectra/'
    specdir = os.path.join(directory,'spectra')
    gc = 'NE_1_002'
    gc_label = 'NE_1_002'   # high metallicity
    gc2 = 'NE_1_003'
    gc2_label = 'NE_1_003'  # low metallicity

    compare = 'M71_J19534827+1848021'
    compare_label = 'M71 J19534827+1848021 [M/H] = -0.7'

    compare2 = 'NGC6791_J19213390+3750202'
    compare_label2 = 'NGC6791 J19213390+3750202 [M/H] = 0.33'


    infiles = ['sme_grid_ne1_002_teff_3447._logg_2.78_feh_-1.00_alpha_0.00_sc_0.00.fits',
               'sme_grid_ne1_002_teff_3447._logg_2.78_feh_-0.70_alpha_0.00_sc_0.00.fits',
               'sme_grid_ne1_002_teff_3447._logg_2.78_feh_0.00_alpha_0.00_sc_0.00.fits',
               'sme_grid_ne1_002_teff_3447._logg_2.78_feh_0.50_alpha_0.00_sc_0.00.fits',
               'sme_grid_ne1_002_teff_3447._logg_2.78_feh_1.00_alpha_0.00_sc_0.00.fits']

    labels = ['-1.0','-0.5','0.0','0.5','1.0']
    indir = '/u/tdo/research/sme_grid/grid/'

    if order == 34:
        v1 = -540.0
        v1a = -520.0
        v2 = -438.5
        v2a = -458.0
        wave_range = [2.2406, 2.271]
        xlim = np.array(wave_range)*1e4
    if order == 35:
        v1 = -240.0
        v1a = -221.0
        v2 = 161.0
        v2a = 141.0
        wave_range = [2.178,2.20882]
        xlim = np.array([2.1880,2.2076])*1e4


    filename = glob.glob(specdir+'/'+gc+'_order'+str(order)+'*.dat')
    gcspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')

    gcspectrum.wavelength = gcspectrum.wavelength*(-v1/3e5+1.0)

    wave = gcspectrum.wavelength.value/1e4
    flux = gcspectrum.flux.value

    filename2 = glob.glob(specdir+'/'+gc2+'_order'+str(order)+'*.dat')

    gcspectrum2 = read_fits_file.read_nirspec_dat(filename2,wave_range=wave_range,desired_wavelength_units='Angstrom')

    gcspectrum2.wavelength = gcspectrum2.wavelength*(-v1a/3e5+1.0)
    wave2 = gcspectrum2.wavelength.value/1e4
    flux2 = gcspectrum2.flux.value


    filename = glob.glob(specdir+'/'+compare2+'_order'+str(order)+'*.dat')
    comparespectrum2 = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')

    #wave = comparespectrum2.wavelength.value/1e4
    #flux = comparespectrum2.flux.value
    #print comparespectrum2.flux[0:10]


    #plot_range = [2.201,2.209]
    plot_range = wave_range
    flux_median = np.median(scipy.stats.sigmaclip(flux)[0])
    flux = flux/flux_median
    plot_abundance_grid(wave,flux,plot_range=plot_range,vel=0,starname=gc_label,
                        infiles=infiles,indir=indir,labels=labels,R=4000.0,
                        label_feh=True,color='grey',min_offset=True)
    #plot_abundance_grid(comparespectrum2.wavelength.value,comparespectrum2.flux.value,vel=v2a)


def plot_abundance_grid(wave,flux,element='Sc',infiles=None,indir=None,labels=None,
                        plot_range=None,vel=None,starname=None,linewidth=2.0,
                        grid_R=50000.0,grid_sampling=14.0,R=20000.0,label_feh=False,
                        min_offset=True,size=12,color='black'):

    # min_offset - minimize the offset between observed and model
    # vel - velocity to move the template spectra

    # by default will plot the comparison with Arcturus grid
    if indir is None:
        indir = '/u/tdo/research/sme_grid/grid/'
    if infiles is None:
        infiles = ['sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_-1.05.fits',
                   'sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_-0.42.fits',
                   'sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_0.05.fits',
                   'sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_0.21.fits',
                   'sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_0.53.fits',
                   'sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_1.00.fits']

    if labels is None:
        labels = ['-1.05','-0.42','0.05','0.21','0.53','1.00']

    if plot_range is None:
        plot_range = [2.205,2.209]
        plot_range = [2.201,2.209]

    if vel is None:
        vel = 0.0  # km/s

    g = operations.InstrumentConvolveGrating(R=R,grid_R=grid_R,grid_sampling=grid_sampling)
    plt.clf()
    # shift the wavelengths
    wave = wave/(vel/3e5+1.0)

    plt.plot(wave,flux,'k',label=starname,linewidth=linewidth)
    plt.xlim(plot_range[0],plot_range[1])

    for i in xrange(len(infiles)):
        tempfile =os.path.join(indir,infiles[i])
        print('plotting: '+tempfile)
        grid_spec = read_fits_file.read_fits_file(tempfile)
        print(np.median(grid_spec.wavelength))
        # measure the radial velocity difference
        grid_wave = grid_spec.wavelength.value/1e4
        grid_flux = grid_spec.flux.value

        #corr = rvmeasure.rvshift(arc_wave,arc_flux,grid_wave,grid_flux)
        #print(corr)

        grid_wave,grid_flux = g.evaluate(grid_wave,grid_flux,R)
        if label_feh:
            temp_label='[M/H] = '+labels[i]
        else:
            temp_label='['+element+'/Fe] = '+labels[i]
        if min_offset:
            offset = fit_spectrum_offset(wave,flux,grid_wave,grid_flux)
        else:
            offset = 0.0
        print offset
        plt.plot(grid_wave,grid_flux-offset,label=temp_label,linewidth=linewidth)
    plt.legend(loc=2,frameon=False,fontsize=size)
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Flux')
    plt.ylim(0.3,1.4)
    plotlines.oplotlines(angstrom=False,arcturus=True,alpha=0.5,color=color)


def fit_spectrum_offset(wave1,flux1,wave2,flux2,interpolate=True):
    # get the best offset between two sets of spectra. wave1 should be
    # the same or have shorter wavlength range than wave2 in order for
    # the interpolation to work

    # interpoplate - interpolate the second spectrum to the same
    # wavelengths as the first before computing offset (default: True)

    # interpolate onto the observed wavelengths
    interp_flux = np.interp(wave1,wave2,flux2)

    popt = scipy.optimize.least_squares(offset_func, 0.0, args = (flux1,interp_flux))
    return popt.x[0]

def offset_func(offset,y1,y2):

    return y1-y2+offset

def test_fit_spectrum_offset():
    gc = 'NE_1_002'
    order = 34
    wave_range = [2.2406, 2.271]
    specdir = os.path.join(directory,'spectra')
    filename = glob.glob(specdir+'/'+gc+'_order'+str(order)+'*.dat')

    gcspectrum = read_fits_file.read_nirspec_dat(filename,desired_wavelength_units='Angstrom',wave_range=wave_range)
    wave = gcspectrum.wavelength.value
    flux = gcspectrum.flux.value
    indir = '/u/tdo/research/sme_grid/grid/'
    infile = 'sme_grid_ne1_002_teff_3447._logg_2.78_feh_-1.00_alpha_0.00_sc_0.00.fits'
    tempfile =os.path.join(indir,infile)
    grid_spec = read_fits_file.read_fits_file(tempfile)
    print(np.median(grid_spec.wavelength))
    # measure the radial velocity difference
    grid_wave = grid_spec.wavelength.value
    grid_flux = grid_spec.flux.value


    fit = fit_spectrum_offset(wave,flux,grid_wave,grid_flux)
    print fit
    plt.clf()
    plt.plot(wave,flux)
    plt.plot(grid_wave,grid_flux+fit)


def test_convolve():


    indir = '/u/tdo/research/sme_grid/grid/'
    filename = os.path.join(indir,'sme_grid_arcturus_teff_4286._logg_1.66_feh_-0.52_alpha_0.20_sc_0.50.fits')
    R = 4000.0
    grid_R = 50000.0
    rescaled_R = 1 / np.sqrt((1/R)**2 - (1 / grid_R)**2 )
    grid_sampling = 71.0
    sigma = ((grid_R / rescaled_R) * grid_sampling /
             (2 * np.sqrt(2 * np.log(2))))
    print(rescaled_R)
    print(sigma)
    g = operations.InstrumentConvolveGrating(R=R,grid_R=grid_R,grid_sampling=grid_sampling)
    grid_spec = read_fits_file.read_fits_file(filename)
    gwave = grid_spec.wavelength.value
    print(gwave[1]-gwave[0])
    print(np.mean(gwave)/(gwave[1]-gwave[0])/500000.0)
    wave, flux = g.evaluate(grid_spec.wavelength.value,grid_spec.flux.value,R)
    plt.clf()
    plt.plot(grid_spec.wavelength.value,grid_spec.flux.value)
    plt.plot(wave,flux)
    plt.xlim(2.1e4,2.17e4)

def find_telluric_rv():
    # compare the telluric to the calibrator to see the initial wavelength
    # offset from the etalons

    #calfile = '/Users/tdo/data/gc/nirspec/20160520/reduce/ngc6819/order33/cal.dat'
    #calfile = '/Users/tdo/data/gc/nirspec/20160520/reduce/m71/order33/cal.dat'
    #calfile = '/Users/tdo/data/gc/nirspec/20160520/reduce/m71/order34/cal.dat'
    calfile = '/group/data/nirspec/20160516/reduce/ngc6791/order34/cal.dat'
    tellfile = '../spectra/telluric_spectrum_1.8-2.5_quinn.txt'
    #wave_range = [2.31, 2.335] # for order 33
    wave_range = [2.24,2.27] # order 34
    calspectrum = read_fits_file.read_nirspec_dat(calfile,
        desired_wavelength_units='Angstrom',wave_range=wave_range,
        single_nod=True)

    calspectrum.flux = rvmeasure.rmcontinuum(calspectrum.wavelength,calspectrum.flux)

    flux_units = 'erg / (cm^2 s Angstrom)'
    wavelength_units = 'Angstrom'

    wavelength,flux = np.loadtxt(tellfile,unpack=True)
    tellrange=[2.2,2.5]
    good = np.where((wavelength >= tellrange[0]) & (wavelength<=tellrange[1]))[0]
    flux = flux[good]
    wavelength=wavelength[good]

    # remove the continuum
    flux = rvmeasure.rmcontinuum(wavelength,flux)

    # shift the wavelengths for a test
    vel = 0.0
    wavelength = (vel/3e5+1.0)*wavelength

    flux = flux * u.Unit(flux_units)
    wavelength = wavelength * 1e4* u.Unit(wavelength_units)
    tellspectrum = Spectrum1D.from_array(wavelength, flux.value, dispersion_unit = wavelength.unit, unit = flux.unit)
    #plt.clf()
    #plt.plot(calspectrum.wavelength,calspectrum.flux)
    #plt.plot(tellspectrum.wavelength,tellspectrum.flux)
    plt.clf()

    #plt.plot(calspectrum.wavelength,calspectrum.flux)
    #plt.plot(tellspectrum.wavelength,tellspectrum.flux)

    corr = rvmeasure.rvshift(calspectrum.wavelength,calspectrum.flux,
        tellspectrum.wavelength,tellspectrum.flux,debug=True,
        lagRange=[-500,500],r1=2e4,r2=2.1e4)
    #print corr

def test_likelihood(g=None):
    # test using chi2 vs. L1 norm for likelihood calculations
    if g is None:
        g=load_grid('/u/tdo/research/phoenix/phoenix_r40000_1.9-2.5_k.h5')

    names = 'NE_1_002'
    #names = 'M71_J19534827+1848021'
    vrad_range = [-500,500]
    R = 30000.0
    order = 35
    snr = 20.0
    wave_ranges = {'order35':[2.178,2.20882],'order34':[2.2406, 2.271]}
    wave_range = wave_ranges['order'+str(order)]

    filename = glob.glob(directory+'spectra/'+names+'_order'+str(order)+'*.dat')
    starspectrum = read_fits_file.read_nirspec_dat(filename[0],wave_range=wave_range,desired_wavelength_units='Angstrom')
    starspectrum.uncertainty = (np.zeros(len(starspectrum.flux.value))+1.0/np.float(snr))*starspectrum.flux.unit
    plt.clf()
    plt.plot(starspectrum.wavelength,starspectrum.flux)

    result2 = fit_star(starspectrum,g,l1norm=True,vrad_range=vrad_range,R=R)
    outname = names+'_order'+str(order)

    outfile2 = 'test_chains/'+outname+'_l1norm.h5'
    result2.to_hdf(outfile2)


    result = fit_star(starspectrum,g,vrad_range=vrad_range,R=R)

    outfile1 = 'test_chains/'+outname+'_chi2.h5'
    result.to_hdf(outfile1)

    r1 = MultiNestResult.from_hdf5(outfile1)
    r2 = MultiNestResult.from_hdf5(outfile2)
    print r1.calculate_sigmas(1)
    print r2.calculate_sigmas(1)

def plot_spectra_compare(primary='NE_1_002',compare = 'NGC6791_J19205+3748282',order=34,
                         primary_label = 'NE1-1 002',
                         compare_label='NGC6791 J19213390+3750202 [M/H] = 0.33',
                         highlight=['Sc','V'],ylim=None,noclear=False,size=14,color='k',
                         offset=0.0,showlines=True,compare_color='C1',spectrum_color='C0',
                         offsets=(),xlim=None):
    # plot spectra from GC stars compared to open clusters

    #directory='/u/tdo/data/gc/nirspec/spectra/'
    specdir = os.path.join(directory,'spectra')
    fitdir = os.path.join('../','nirspec_fits')
    gc = 'NE_1_002'
    gc_label = 'NE_1_002 [M/H]'
    gc2 = 'NE_1_003'
    gc2_label = 'NE_1_003 [M/H] < -1.0'

    if order == 34:
        wave_range = [2.2406, 2.271]
        if xlim is None:
            xlim = np.array([2.2445,2.268])*1e4
        #xlim = np.array(wave_range)*1e4
        vel_str = 'vrad_3'
    if order == 35:
        wave_range = [2.178,2.20882]
        if xlim is None:
            xlim = np.array([2.1804,2.20882])*1e4
        vel_str = 'vrad_4'
    if order == 36:
        wave_range = [2.11695,2.14566]
        if xlim is None:
            xlim = np.array(wave_range)*1e4
        vel_str='vrad_5'
    if order == 37:
        wave_range = [2.0750,2.0875]
        if xlim is None:
            xlim = np.array(wave_range)*1e4
        vel_str='vrad_6'
        

    primary_file = os.path.join(fitdir,primary+'_results.h5')
    primary_results = MultiNestResult.from_hdf5(primary_file)
    v1 = primary_results.maximum[vel_str]
    print primary_results.maximum
    compare_file = os.path.join(fitdir,compare+'_results.h5')
    compare_results = MultiNestResult.from_hdf5(compare_file)
    v2 = compare_results.maximum[vel_str]
    print compare_results.maximum    
    print 'velocities: ',v1,v2
    filename = glob.glob(specdir+'/'+primary+'_order'+str(order)+'*.dat')
    gcspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
    gcspectrum.wavelength = gcspectrum.wavelength*(-v1/3e5+1.0)

    filename2 = glob.glob(specdir+'/'+compare+'_order'+str(order)+'*.dat')
    comparespectrum = read_fits_file.read_nirspec_dat(filename2,wave_range=wave_range,desired_wavelength_units='Angstrom')
    comparespectrum.wavelength = comparespectrum.wavelength*(-v2/3e5+1.0)

    if not noclear:
        plt.clf()
    plt.plot(gcspectrum.wavelength,gcspectrum.flux.value+offset,label=primary_label,color=spectrum_color)
    plt.plot(comparespectrum.wavelength,comparespectrum.flux.value+offset,
             label=compare_label,color=compare_color)
    plt.legend(loc=2,fontsize=size)

    plt.xlim(*xlim)
    if ylim is None:
        plt.ylim(0.3,1.3)
    else:
        plt.ylim(*ylim)
    if showlines:
        plotlines.oplotlines(angstrom=True,arcturus=True,alpha=0.5,highlight=highlight,
                             size=size,color=color,molecules=False,offsets=offsets)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux')
    
def plot_nirspec_paper(offsets=()):
    '''
    Make plots for the NIRSPEC high-metallicity paper
    
    '''

    star1 = 'NE_1_001'
    star1_label = 'GC Star: NE1-1 001'
    
    star2 = 'NE_1_002'
    star2_label = 'GC Star: NE1-1 002'

    compare = 'NGC6791_J19205+3748282'
    compare_label = 'Calibrator Star: J19213390+3750202'
    size=12
    highlight = ['Sc','V','Y']
    offset = 0.5
    ylim = [0.3,2.0]
    plt.clf()
    #plt.figure(figsize=(12,6))

    wave1 = [21804.6,21909.8]
    wave2 = [21996.2,22078.8]
    wave3 = [22462.1,22556.4]
    wave4 = [22611.0,22669.7]
    
    plt.subplot(2,2,1)
    plot_spectra_compare(primary=star1,primary_label=star1_label,compare=compare,
                         compare_label= None, order=35,noclear=True,
                         size=size,color='grey',highlight=highlight,
                         showlines=False,spectrum_color='k',offsets=offsets,xlim=wave1)    

    plot_spectra_compare(primary=star2,primary_label=star2_label,compare=compare,
                         compare_label=compare_label, order=35,noclear=True,spectrum_color='C0',
                         size=size,color='grey',highlight=highlight,offset=offset,
                         ylim=ylim,showlines=True,compare_color='C1',offsets=offsets,xlim=wave1)

    plt.subplot(2,2,2)
    plot_spectra_compare(primary=star1,primary_label=None,compare=compare,
                         compare_label= None, order=35,noclear=True,
                         size=size,color='grey',highlight=highlight,
                         showlines=False,spectrum_color='k',offsets=offsets,xlim=wave2)    

    plot_spectra_compare(primary=star2,primary_label=None,compare=compare,
                         compare_label=None, order=35,noclear=True,spectrum_color='C0',
                         size=size,color='grey',highlight=highlight,offset=offset,
                         ylim=ylim,showlines=True,compare_color='C1',offsets=offsets,xlim=wave2)

    #plt.savefig('../plots/nirspec_empirical_compare_order35.pdf')


    plt.subplot(2,2,3)
    plot_spectra_compare(primary=star1,primary_label=None,compare=compare,
                         compare_label= None, order=34,noclear=True,spectrum_color='k',
                         size=size,color='grey',highlight=highlight,showlines=False,
                         offsets=offsets,xlim=wave3)    

    plot_spectra_compare(primary=star2,primary_label=None,compare=compare,
                         compare_label=None, order=34,noclear=True,spectrum_color='C0',
                         size=size,color='grey',highlight=highlight,offset=offset,ylim=ylim,
                         offsets=offsets,xlim=wave3)

    plt.subplot(2,2,4)
    plot_spectra_compare(primary=star1,primary_label=None,compare=compare,
                         compare_label= None, order=34,noclear=True,spectrum_color='k',
                         size=size,color='grey',highlight=highlight,showlines=False,
                         offsets=offsets,xlim=wave4)    

    plot_spectra_compare(primary=star2,primary_label=None,compare=compare,
                         compare_label=None, order=34,noclear=True,spectrum_color='C0',
                         size=size,color='grey',highlight=highlight,offset=offset,ylim=ylim,
                         offsets=offsets,xlim=wave4)
    
    #plt.savefig('../plots/nirspec_empirical_compare_order34.pdf')
    plt.savefig('../plots/nirspec_empirical_compare.pdf')
    ## plt.clf()
    ## plt.subplot(2,1,1)
   
    ## plot_spectra_compare(primary=star1,primary_label=star1_label,compare=compare,
    ##                      compare_label= compare_label, order=36,noclear=True,
    ##                      size=size,color='grey',highlight=highlight)    
    ## plt.subplot(2,1,2)
    ## plot_spectra_compare(primary=star2,primary_label=star2_label,compare=compare,
    ##                      compare_label=compare_label, order=36,noclear=True,
    ##                      size=size,color='grey',highlight=highlight)
    ## plt.savefig('../plots/nirspec_empirical_compare_order36.pdf')

    ## plt.clf()
    ## plt.subplot(2,1,1)
   
    ## plot_spectra_compare(primary=star1,primary_label=star1_label,compare=compare,
    ##                      compare_label= compare_label, order=37,noclear=True,
    ##                      size=size,color='grey',highlight=highlight)    
    ## plt.subplot(2,1,2)
    ## plot_spectra_compare(primary=star2,primary_label=star2_label,compare=compare,
    ##                      compare_label=compare_label, order=37,noclear=True,
    ##                      size=size,color='grey',highlight=highlight)
    ## plt.savefig('../plots/nirspec_empirical_compare_order37.pdf')
    
def mk_results_table(directory='../nirspec_fits/'):

    # make the results table with the best fit models
    starnames = ['NE1-1 001', 'NE1-1 002', 'NGC6791 J19213390+3750202']

    results_files = ['NE_1_001_results.h5','NE_1_002_results.h5','NGC6791_J19205+3748282_results.h5']
    teff =[]
    teff_err = []
    logg = []
    logg_err =[]
    mh = []
    mh_err =[]
    alpha = []
    alpha_err = []
    for i in np.arange(len(results_files)):
        hd5file = os.path.join(directory,results_files[i])
        result = MultiNestResult.from_hdf5(hd5file)
        m = result.median
        sig = result.calculate_sigmas(1)
        print sig
        teff.append('%4.f' % m['teff_0'])
        #teff_err.append(r'^{+%3.f}_{-%3.f}' % (sig['teff_0'][0]-m['teff_0'],m['teff_0']-sig['teff_0'][1]))
        teff_err.append(r'%3.f' % np.abs(sig['teff_0'][0]-sig['teff_0'][1]))
        logg.append('%3.1f' % m['logg_0'])
        mh.append('%3.2f' % m['mh_0'])
        alpha.append('%3.2f' % m['alpha_0'])

    d = collections.OrderedDict([('Name',starnames),('Teff',teff),(r'$\sigma_{T_{eff}}$',teff_err),
                                 ('log g',logg),('[M/H]',mh),('[alpha/Fe]',alpha)])
    p = pd.DataFrame(d)
    print p.to_latex(index=False,escape=False)

def mk_summary_table(directory='../nirspec_fits/'):
    # make the results table with the best fit models
    starnames = ['NE1-1 001', 'NE1-1 002', 'NGC6791 J19213390+3750202']

    results_files = ['NE_1_001_results.h5','NE_1_002_results.h5','NGC6791_J19205+3748282_results.h5']
    teff =[]
    teff_err = []
    logg = []
    logg_err =[]
    mh = []
    mh_err =[]
    alpha = []
    alpha_err = []
    for i in np.arange(len(results_files)):
        hd5file = os.path.join(directory,results_files[i])
        result = MultiNestResult.from_hdf5(hd5file)
        m = result.maximum
        med = result.median
        sig = result.calculate_sigmas(1)
        print(results_files[i])
        print('Param\t Max \t Median \t Lower \t Upper \t sigma')
        for k in sig.keys():
            print('%s\t %f\t %f\t %f\t %f\t %f' % (k,m[k],med[k],sig[k][0],sig[k][1],(sig[k][1]-sig[k][0])/2.0))
