import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
import math
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
import operator
import sys
from matplotlib.pyplot import cm
import multi_order_fitting_functions as mtf
from scipy.stats.stats import pearsonr
from multi_order_fitting_functions import Splitter3, Combiner3, Splitter4, Combiner4
from scipy.optimize import curve_fit
from scipy import ndimage as nd


def getKey(item):
    return item[0]

def load_full_grid_phoenix():
    g = load_grid('/u/ghezgroup/data/metallicity/nirspec/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    return g

def load_full_grid_bosz():
    g = load_grid('/u/ghezgroup/data/metallicity/nirspec/grids/bosz_t3500_7000_w20000_24000_R25000.h5')
    return g

def sl_response_plot_three(starname,g,specdir='/group/data/nirspec/spectra/',snr=30.,nnorm=2):

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

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/mh_masked*_bosz.h5')

    cut_lis = []

    for filename in h5_files_us:
        print filename.split('_')
        cut_lis += [(float(filename.split('_')[6]),filename)]

    cut_lis = sorted(cut_lis,key = getKey)

    h5_files = [i[1] for i in cut_lis]

    sl_val = []
    print h5_files
    for filename in h5_files:
        print filename.split('_')[6]
        gc_result = MultiNestResult.from_hdf5(filename)
        print gc_result


        sl_mh1,sl_mh2,sl_mh3 = mtf.s_lambda_three_order(model,'mh',model.mh_0.value,0.1)


        w1,f1,w2,f2,w3,f3 = model()
        #combine all sl_mh,w,f, lists

        sl_mh = np.concatenate((sl_mh1,sl_mh2,sl_mh3))

        w = np.concatenate((w1,w2,w3))

        f = np.concatenate((f1,f2,f3))

        starfluxall = np.concatenate((starspectrum34.flux.value,starspectrum35.flux.value,starspectrum36.flux.value))

        starwaveall = np.concatenate((starspectrum34.wavelength.value,starspectrum35.wavelength.value,starspectrum36.wavelength.value))

        sigma_bounds = gc_result.calculate_sigmas(1)

        sigmas = []

        for a in sigma_bounds.keys():
            print a
            sigmas += [(sigma_bounds[a][1]-sigma_bounds[a][0])/2.]

        print sigmas

        abs_sl_mh = []
        
        mask_sl_f = []
        mask_sl_w = []

    
        data_sl_f = []

        for i in range(len(sl_mh)):
            abs_sl_mh += [np.abs(sl_mh[i])]
            if abs(sl_mh[i]) < float(filename.split('_')[6]):
                mask_sl_f += [starfluxall[i]]
                mask_sl_w += [starwaveall[i]]
            else:
                data_sl_f += [starfluxall[i]]

        print sigmas
        sl_val += [(float(filename.split('_')[6]),len(mask_sl_f),gc_result.median['vrad_3'],gc_result.median['vrad_4'],gc_result.median['vrad_5'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],gc_result.median['teff_0'],sigmas)]

    print len(starfluxall)
    return sl_val


def plot_sl_response(sl_val,starname, plot_ref_vals=False,log_scale=False, invert_xaxes=False):
    f= plt.figure(figsize=(12,11))
    rvax  = f.add_subplot(6,1,1)
    loggax  = f.add_subplot(6,1,2)
    mhax  = f.add_subplot(6,1,3)
    alphaax = f.add_subplot(6,1,4)
    teffax = f.add_subplot(6,1,5)
    lenax = f.add_subplot(6,1,6)

    
    sl_val = sorted(sl_val)
    #ax = plt.subplot()
    print [i[1] for i in sl_val]
    print [i[0] for i in sl_val]
    print len(sl_val)
    
    rvax.errorbar([i[0] for i in sl_val],[i[2] for i in sl_val],[i[10][5] for i in sl_val], color='red',fmt='o',linestyle='--',capsize=5,label='Order 34')
    rvax.errorbar([i[0] for i in sl_val],[i[3] for i in sl_val],[i[10][6] for i in sl_val], color='green',fmt='o',linestyle='--',capsize=5,label='Order 35')
    rvax.errorbar([i[0] for i in sl_val],[i[4] for i in sl_val],[i[10][7] for i in sl_val], color='blue',fmt='o',linestyle='--',capsize=5,label='Order 36')
    rvax.errorbar([i[0] for i in sl_val],[i[5] for i in sl_val],[i[10][8] for i in sl_val], fmt='o',linestyle='--',capsize=5,label='Order 37')


    rvax.set_ylabel('Radial Velocity (km/s)')
    rvax.set_title(starname+' masked fit responses')

    loggax.errorbar([i[0] for i in sl_val],[i[6] for i in sl_val],[i[10][1] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    loggax.set_ylabel('Log g')

    mhax.errorbar([i[0] for i in sl_val],[i[7] for i in sl_val],[i[10][2] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    mhax.set_ylabel('[M/H]')

    alphaax.errorbar([i[0] for i in sl_val],[i[8] for i in sl_val],[i[10][3] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    alphaax.set_ylabel('$alpha$')

    teffax.errorbar([i[0] for i in sl_val],[i[9] for i in sl_val],[i[10][0] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    teffax.set_ylabel('$T_{eff}$ (K)')

    lenax.plot([i[0] for i in sl_val],[i[1] for i in sl_val], color='black')
    lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')
    lenax.set_xlabel('$S_{\lambda}$ cutoff')
    #lenax.set_xlabel('Residual cutoff')
    lenax.set_ylabel('# points masked')
    lenax.legend(bbox_to_anchor=(0.95,0.75), fontsize=13)

    if log_scale:
        rvax.set_xscale('log')
        loggax.set_xscale('log')
        mhax.set_xscale('log')
        alphaax.set_xscale('log')
        teffax.set_xscale('log')
        lenax.set_xscale('log')

    if invert_xaxes:
        rvax.invert_xaxis()
        loggax.invert_xaxis()
        mhax.invert_xaxis()
        alphaax.invert_xaxis()
        teffax.invert_xaxis()
        lenax.invert_xaxis()

    if plot_ref_vals:
        cal_star_info = scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None)
        cal_star_info = map(list, zip(*cal_star_info))
        if starname in cal_star_info[0]:
            print starname+' is a calibrator star'
            star_ind = cal_star_info[0].index(starname)
            rvax.axhline(y=cal_star_info[5][star_ind], color='y',linestyle='--',label='APOGEE value')
            mhax.axhline(y=cal_star_info[1][star_ind], color='y',linestyle='--')
            teffax.axhline(y=cal_star_info[2][star_ind], color='y',linestyle='--')
            alphaax.axhline(y=cal_star_info[4][star_ind], color='y',linestyle='--')
            loggax.axhline(y=cal_star_info[3][star_ind], color='y',linestyle='--')


        
    rvax.legend(bbox_to_anchor=(0.95,0.75), fontsize=13)
    plt.show()


def sl_masked_param_info(starname,g,specdir='/group/data/nirspec/spectra/',snr=30.,nnorm=2):

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
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm4 = Normalize(starspectrum37,nnorm)



    model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
         norm1 & norm2 & norm3 & norm4

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/masked_sl*'+starname+'_order34-37_bosz.h5')

    cut_lis = []

    for filename in h5_files_us:
        cut_lis += [(float(filename.split('_')[6]),filename)]

    cut_lis = sorted(cut_lis,key = getKey)

    h5_files = [i[1] for i in cut_lis]
    
    sl_val = []

    combined_data_mask_model = {}
    

    for filename in h5_files:
        gc_result = MultiNestResult.from_hdf5(filename)
        print gc_result

        for a in apogee_vals.keys():
            setattr(model,a,apogee_vals[a])

        w1,f1,w2,f2,w3,f3,w4,f4 = model()

        sl_mh1,sl_mh2,sl_mh3,sl_mh4 = mtf.s_lambda_four_order(model,'mh',model.mh_0.value,0.1)



        #combine all sl_mh,w,f, lists

        sl_mh = np.concatenate((sl_mh1,sl_mh2,sl_mh3,sl_mh4))

        #w = np.concatenate((w1/(gc_result.median['vrad_3']/3e5+1.0),w2/(gc_result.median['vrad_4']/3e5+1.0),w3/(gc_result.median['vrad_5']/3e5+1.0),w4/(gc_result.median['vrad_6']/3e5+1.0)))

        w = np.concatenate((w1,w2,w3,w4))

        
        f = np.concatenate((f1,f2,f3,f4))

        starfluxall = np.concatenate((starspectrum34.flux.value,starspectrum35.flux.value,starspectrum36.flux.value,starspectrum37.flux.value))

        starwaveall = np.concatenate((starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0)))

        sigma_bounds = gc_result.calculate_sigmas(1)

        sigmas = []

        for a in sigma_bounds.keys():
            #print a
            sigmas += [(sigma_bounds[a][1]-sigma_bounds[a][0])/2.]

        #print sigmas

        abs_sl_mh = []
        
        mask_sl_f = []
        mask_sl_w = []

    
        data_sl_f = []

        for i in range(len(sl_mh)):
            abs_sl_mh += [np.abs(sl_mh[i])]
            if abs(sl_mh[i]) < float(filename.split('_')[6]):
                mask_sl_f += [starfluxall[i]]
                mask_sl_w += [starwaveall[i]]
            else:
                data_sl_f += [starfluxall[i]]

        #combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})

        combined_data_mask_model.update({filename.split('_')[6] : [(starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value),\
                                                                   (starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value),\
                                                                   (starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value),\
                                                                   (starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0),starspectrum37.flux.value),\
                                                                   (w1,f1),(w2,f2),(w3,f3),(w4,f4),(mask_sl_w,mask_sl_f)]})
        
        sl_val += [(float(filename.split('_')[6]),len(mask_sl_f),gc_result.median['vrad_3'],gc_result.median['vrad_4'],gc_result.median['vrad_5'],gc_result.median['vrad_6'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],gc_result.median['teff_0'],sigmas)]

    return sl_val, combined_data_mask_model


def sl_masked_param_info_three_order(starname, g, specdir='/group/data/nirspec/spectra/', snr=30., nnorm=2):
    file1 = glob.glob(specdir + starname + '_order34*.dat')
    file2 = glob.glob(specdir + starname + '_order35*.dat')
    file3 = glob.glob(specdir + starname + '_order36*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum34.flux.unit

    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
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

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl*' + starname + '_order34-36_phoenix.h5')

    cut_lis = []

    for filename in h5_files_us:
        cut_lis += [(float(filename.split('_')[7]), filename)]

    cut_lis = sorted(cut_lis, key=getKey)

    h5_files = [i[1] for i in cut_lis]

    sl_val = []

    combined_data_mask_model = {}

    for filename in h5_files:
        gc_result = MultiNestResult.from_hdf5(filename)
        print gc_result

        unmasked_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_0.0_' + starname + '_order34-36.h5')

        for a in unmasked_result.median.keys():
            setattr(model,a,unmasked_result.median[a])

        model = mtf.correct_bounds(model)

        w1, f1, w2, f2, w3, f3 = model()

        print f1[0:30],f2[0:30],f3[0:30]

        print model

        sl_mh1, sl_mh2, sl_mh3 = mtf.s_lambda_three_order(model, 'mh', model.mh_0.value, 0.1)

        # combine all sl_mh,w,f, lists

        sl_mh = np.concatenate((sl_mh1, sl_mh2, sl_mh3))

        # w = np.concatenate((w1/(gc_result.median['vrad_3']/3e5+1.0),w2/(gc_result.median['vrad_4']/3e5+1.0),w3/(gc_result.median['vrad_5']/3e5+1.0),w4/(gc_result.median['vrad_6']/3e5+1.0)))

        w = np.concatenate((w1, w2, w3))

        f = np.concatenate((f1, f2, f3))

        starfluxall = np.concatenate((starspectrum34.flux.value, starspectrum35.flux.value, starspectrum36.flux.value))

        starwaveall = np.concatenate((starspectrum34.wavelength.value / (gc_result.median['vrad_3'] / 3e5 + 1.0),
                                      starspectrum35.wavelength.value / (gc_result.median['vrad_4'] / 3e5 + 1.0),
                                      starspectrum36.wavelength.value / (gc_result.median['vrad_5'] / 3e5 + 1.0)))

        sigma_bounds = gc_result.calculate_sigmas(1)

        sigmas = []

        for a in sigma_bounds.keys():
            # print a
            sigmas += [(sigma_bounds[a][1] - sigma_bounds[a][0]) / 2.]

        # print sigmas

        abs_sl_mh = []

        mask_sl_f = []
        mask_sl_w = []

        data_sl_f = []

        for i in range(len(sl_mh)):
            abs_sl_mh += [np.abs(sl_mh[i])]
            if abs(sl_mh[i]) < float(filename.split('_')[7]):
                mask_sl_f += [starfluxall[i]]
                mask_sl_w += [starwaveall[i]]
            else:
                data_sl_f += [starfluxall[i]]

        # combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})

        combined_data_mask_model.update({filename.split('_')[6]: [
            (starspectrum34.wavelength.value / (gc_result.median['vrad_3'] / 3e5 + 1.0), starspectrum34.flux.value), \
            (starspectrum35.wavelength.value / (gc_result.median['vrad_4'] / 3e5 + 1.0), starspectrum35.flux.value), \
            (starspectrum36.wavelength.value / (gc_result.median['vrad_5'] / 3e5 + 1.0), starspectrum36.flux.value), \
            (w1, f1), (w2, f2), (w3, f3), (mask_sl_w, mask_sl_f)]})

        sl_val += [(float(filename.split('_')[7]), len(mask_sl_f), gc_result.median['vrad_3'],
                    gc_result.median['vrad_4'], gc_result.median['vrad_5'],
                    gc_result.median['logg_0'], gc_result.median['mh_0'], gc_result.median['alpha_0'],
                    gc_result.median['teff_0'], sigmas)]

    return sl_val, combined_data_mask_model


def residual_masked_param_info_four_order(starname,g,specdir='/group/data/nirspec/spectra/',snr=30.,nnorm=2):

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
    #rot3 = RotationalBroadening.from_grid(g,vrot=np.array([10.0]))
    norm4 = Normalize(starspectrum37,nnorm)



    model = g | rot1 | Splitter4() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
         convolve1 & convolve2 & convolve3 & convolve4 | interp1 & interp2 & interp3 & interp4 | \
         norm1 & norm2 & norm3 & norm4


    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/masked_res*'+starname+'_order34-37.h5')

    cut_lis = []

    for filename in h5_files_us:
        print filename.split('_')
        cut_lis += [(float(filename.split('_')[6]),filename)]

    cut_lis = sorted(cut_lis,key = getKey)

    h5_files = [i[1] for i in cut_lis]
    
    res_val = []

    combined_data_mask_model = {}
    

    for filename in h5_files:
        gc_result = MultiNestResult.from_hdf5(filename)

        for a in apogee_vals.keys():
            setattr(model,a,apogee_vals[a])

        w1,f1,w2,f2,w3,f3,w4,f4 = model()

        
        w = np.concatenate((w1,w2,w3,w4))
        
        f = np.concatenate((f1,f2,f3,f4))

        starfluxall = np.concatenate((starspectrum34.flux.value,starspectrum35.flux.value,starspectrum36.flux.value,starspectrum37.flux.value))

        starwaveall = np.concatenate((starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0)))

        residual_flux34 = mt.calc_residuals(f1,starspectrum34.flux.value)
        residual_flux35 = mt.calc_residuals(f2,starspectrum35.flux.value)
        residual_flux36 = mt.calc_residuals(f3,starspectrum36.flux.value)
        residual_flux37 = mt.calc_residuals(f4,starspectrum37.flux.value)

        residual_flux_all = np.concatenate((residual_flux34,residual_flux35,residual_flux36,residual_flux37))

        residual_masked_flux = []
        residual_masked_wavelength = []

        for i in range(len(residual_flux_all)):
            if residual_flux_all[i] > float(filename.split('_')[6]):
                residual_masked_flux += [starfluxall[i]]
                residual_masked_wavelength += [starwaveall[i]]
                                            
        sigma_bounds = gc_result.calculate_sigmas(1)

        sigmas = []

        for a in sigma_bounds.keys():
            #print a
            sigmas += [(sigma_bounds[a][1]-sigma_bounds[a][0])/2.]

        #print sigmas

        #combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})

        combined_data_mask_model.update({filename.split('_')[6] : [(starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value),\
                                                                   (starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value),\
                                                                   (starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value),\
                                                                   (starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0),starspectrum37.flux.value),\
                                                                   (w1,f1),(w2,f2),(w3,f3),(w4,f4),(residual_masked_wavelength,residual_masked_flux)]})
        
        res_val += [(float(filename.split('_')[6]),len(residual_masked_flux),gc_result.median['vrad_3'],gc_result.median['vrad_4'],gc_result.median['vrad_5'],gc_result.median['vrad_6'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],gc_result.median['teff_0'],sigmas)]
         
    print len(starfluxall)
    return res_val, combined_data_mask_model


def residual_masked_param_info_three_order(starname, g, specdir='/group/data/nirspec/spectra/', snr=30., nnorm=2):
    file1 = glob.glob(specdir + starname + '_order34*.dat')
    file2 = glob.glob(specdir + starname + '_order35*.dat')
    file3 = glob.glob(specdir + starname + '_order36*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom', wave_range=waverange34)
    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum34.flux.unit

    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom', wave_range=waverange35)
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum35.flux.unit

    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom', wave_range=waverange36)
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

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/masked_res*' + starname + '_order34-36_phoenix.h5')

    cut_lis = []

    for filename in h5_files_us:
        print filename.split('_')
        cut_lis += [(float(filename.split('_')[6]), filename)]

    cut_lis = sorted(cut_lis, key=getKey)

    h5_files = [i[1] for i in cut_lis]

    res_val = []

    combined_data_mask_model = {}

    for filename in h5_files:
        gc_result = MultiNestResult.from_hdf5(filename)

        unmasked_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_0.0_' + starname + '_order34-36.h5')

        for a in unmasked_result.median.keys():
            setattr(model,a,unmasked_result.median[a])

        model = mtf.correct_bounds(model)

        w1, f1, w2, f2, w3, f3 = model()

        w = np.concatenate((w1, w2, w3))

        f = np.concatenate((f1, f2, f3))

        starfluxall = np.concatenate((starspectrum34.flux.value, starspectrum35.flux.value, starspectrum36.flux.value))

        starwaveall = np.concatenate((starspectrum34.wavelength.value / (gc_result.median['vrad_3'] / 3e5 + 1.0),
                                      starspectrum35.wavelength.value / (gc_result.median['vrad_4'] / 3e5 + 1.0),
                                      starspectrum36.wavelength.value / (gc_result.median['vrad_5'] / 3e5 + 1.0)))

        residual_flux34 = mt.calc_residuals(f1, starspectrum34.flux.value)
        residual_flux35 = mt.calc_residuals(f2, starspectrum35.flux.value)
        residual_flux36 = mt.calc_residuals(f3, starspectrum36.flux.value)

        residual_flux_all = np.concatenate((residual_flux34, residual_flux35, residual_flux36))

        residual_masked_flux = []
        residual_masked_wavelength = []

        for i in range(len(residual_flux_all)):
            if residual_flux_all[i] > float(filename.split('_')[6]):
                residual_masked_flux += [starfluxall[i]]
                residual_masked_wavelength += [starwaveall[i]]

        sigma_bounds = gc_result.calculate_sigmas(1)

        sigmas = []

        for a in sigma_bounds.keys():
            # print a
            sigmas += [(sigma_bounds[a][1] - sigma_bounds[a][0]) / 2.]

        # print sigmas

        # combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})

        combined_data_mask_model.update({filename.split('_')[6]: [
            (starspectrum34.wavelength.value / (gc_result.median['vrad_3'] / 3e5 + 1.0), starspectrum34.flux.value), \
            (starspectrum35.wavelength.value / (gc_result.median['vrad_4'] / 3e5 + 1.0), starspectrum35.flux.value), \
            (starspectrum36.wavelength.value / (gc_result.median['vrad_5'] / 3e5 + 1.0), starspectrum36.flux.value), \
            (w1, f1), (w2, f2), (w3, f3), (residual_masked_wavelength, residual_masked_flux)]})

        res_val += [(float(filename.split('_')[6]), len(residual_masked_flux), gc_result.median['vrad_3'],
                     gc_result.median['vrad_4'], gc_result.median['vrad_5'],
                     gc_result.median['logg_0'], gc_result.median['mh_0'], gc_result.median['alpha_0'],
                     gc_result.median['teff_0'], sigmas)]

    print len(starfluxall)
    return res_val, combined_data_mask_model


def plot_sl_res_response_four_order(sl_val,res_val,starname, savefig=False):
    f= plt.figure(figsize=(24,12))

    
    sl_rvax  = f.add_subplot(6,2,1)
    sl_loggax  = f.add_subplot(6,2,3)
    sl_mhax  = f.add_subplot(6,2,5)
    sl_alphaax = f.add_subplot(6,2,7)
    sl_teffax = f.add_subplot(6,2,9)
    sl_lenax = f.add_subplot(6,2,11)


    res_rvax  = f.add_subplot(6,2,2)
    res_loggax  = f.add_subplot(6,2,4)
    res_mhax  = f.add_subplot(6,2,6)
    res_alphaax = f.add_subplot(6,2,8)
    res_teffax = f.add_subplot(6,2,10)
    res_lenax = f.add_subplot(6,2,12)


    allax = f.get_axes()
    
    sl_val = sorted(sl_val)
    res_val = sorted(res_val)

    

    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))
    print [x[0] for x in cal_star_info]
    if starname in [x[0] for x in cal_star_info]:
        print starname+' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]

    order_info = scipy.genfromtxt('/u/rbentley/metallicity/reduced_orders.dat', delimiter='\t', skip_header=1,dtype=None)

    print sl_val[0][0]
    
    unmasked_rv = None
    if sl_val[0][0] == 0.0:
        unmasked_rv = (sl_val[0][2],sl_val[0][3],sl_val[0][4],sl_val[0][5])
    else:
        unmasked_results = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/masked_sl_cutoff_0.0_'+starname+'order34-37.h5')
        unmasked_rv = (unmasked_results.median['vrad_3'],unmasked_results.median['vrad_4'],unmasked_results.median['vrad_5'],unmasked_results.median['vrad_6'])
    
    sl_rvax.errorbar([i[0] for i in sl_val],[i[2]-unmasked_rv[0] for i in sl_val],[i[10][5] for i in sl_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
    sl_rvax.errorbar([i[0] for i in sl_val],[i[3]-unmasked_rv[1] for i in sl_val],[i[10][6] for i in sl_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
    sl_rvax.errorbar([i[0] for i in sl_val],[i[4]-unmasked_rv[2] for i in sl_val],[i[10][7] for i in sl_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
    sl_rvax.errorbar([i[0] for i in sl_val],[i[5]-unmasked_rv[3] for i in sl_val],[i[10][8] for i in sl_val], fmt='o',linestyle='--',capsize=5,label='Order 37')
    sl_rvax.set_ylabel('Radial Velocity difference (km/s)',fontsize=8)
    sl_rvax.set_title(starname+' $S_{\lambda}$ masked fit responses',fontsize=12)

    sl_loggax.errorbar([i[0] for i in sl_val],[i[6]-float(cal_star_info[3]) for i in sl_val],[i[10][1] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    sl_loggax.set_ylabel('Log g difference',fontsize=8)

    sl_mhax.errorbar([i[0] for i in sl_val],[i[7]-float(cal_star_info[1]) for i in sl_val],[i[10][2] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    sl_mhax.set_ylabel('[M/H] difference',fontsize=8)

    sl_alphaax.errorbar([i[0] for i in sl_val],[i[8]-float(cal_star_info[4]) for i in sl_val],[i[10][3] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    sl_alphaax.set_ylabel('$alpha$ difference',fontsize=8)

    sl_teffax.errorbar([i[0] for i in sl_val],[i[9]-float(cal_star_info[2]) for i in sl_val],[i[10][0] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    sl_teffax.set_ylabel('$T_{eff}$ (K) difference',fontsize=8)

    sl_lenax.plot([i[0] for i in sl_val],[i[1] for i in sl_val], color='black')
    sl_lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')

    sl_lenax.set_xlabel('$S_{\lambda}$ cutoff',fontsize=8)
    sl_lenax.set_ylabel('# points masked',fontsize=8)
    #sl_lenax.legend(loc='center right', fontsize=13)

    sl_rvax.set_xscale('log')
    sl_loggax.set_xscale('log')
    sl_mhax.set_xscale('log')
    sl_alphaax.set_xscale('log')
    sl_teffax.set_xscale('log')
    sl_lenax.set_xscale('log')


    res_rvax.errorbar([i[0] for i in res_val],[i[2]-unmasked_rv[0] for i in res_val],[i[10][5] for i in res_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
    res_rvax.errorbar([i[0] for i in res_val],[i[3]-unmasked_rv[1] for i in res_val],[i[10][6] for i in res_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
    res_rvax.errorbar([i[0] for i in res_val],[i[4]-unmasked_rv[2] for i in res_val],[i[10][7] for i in res_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
    res_rvax.errorbar([i[0] for i in res_val],[i[5]-unmasked_rv[3] for i in res_val],[i[10][8] for i in res_val], fmt='o',linestyle='--',capsize=5,label='Order 37')

    
    res_rvax.set_ylabel('Radial Velocity difference (km/s)',fontsize=8)
    res_rvax.set_title(starname+' residual masked fit responses',fontsize=12)

    res_loggax.errorbar([i[0] for i in res_val],[i[6]-float(cal_star_info[3]) for i in res_val],[i[10][1] for i in res_val], color='black',fmt='o',linestyle='--',capsize=5)
    res_loggax.set_ylabel('Log g difference',fontsize=8)

    res_mhax.errorbar([i[0] for i in res_val],[i[7]-float(cal_star_info[1]) for i in res_val],[i[10][2] for i in res_val], color='black',fmt='o',linestyle='--',capsize=5)
    res_mhax.set_ylabel('[M/H] difference',fontsize=8)

    res_alphaax.errorbar([i[0] for i in res_val],[i[8]-float(cal_star_info[4]) for i in res_val],[i[10][3] for i in res_val], color='black',fmt='o',linestyle='--',capsize=5)
    res_alphaax.set_ylabel('$alpha$ difference',fontsize=8)

    res_teffax.errorbar([i[0] for i in res_val],[i[9]-float(cal_star_info[2]) for i in res_val],[i[10][0] for i in res_val], color='black',fmt='o',linestyle='--',capsize=5)
    res_teffax.set_ylabel('$T_{eff}$ difference(K)',fontsize=8)

    res_lenax.plot([i[0] for i in res_val],[i[1] for i in res_val], color='black')
    res_lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')
    res_lenax.set_xlabel('Residual cutoff',fontsize=12)
    res_lenax.set_ylabel('# points masked',fontsize=12)
    res_lenax.legend(loc='center right', fontsize=13)
    
    res_rvax.invert_xaxis()
    res_loggax.invert_xaxis()
    res_mhax.invert_xaxis()
    res_alphaax.invert_xaxis()
    res_teffax.invert_xaxis()
    res_lenax.invert_xaxis()

    for ax in allax:
        ax.axhline(y=0.0, color='y',linestyle='--',label='APOGEE value')

        
    res_rvax.legend(bbox_to_anchor=(0.95,0.75), fontsize=10)

    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    plt.rc('axes',labelsize=12)
    if savefig:
        plt.savefig('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/sl-res_response_plot_'+starname+'.png')
    plt.show()


def plot_sl_res_response_three_order(sl_val, res_val, starname, savefig=False):
    f = plt.figure(figsize=(24, 12))

    sl_rvax = f.add_subplot(6, 2, 1)
    sl_loggax = f.add_subplot(6, 2, 3)
    sl_mhax = f.add_subplot(6, 2, 5)
    sl_alphaax = f.add_subplot(6, 2, 7)
    sl_teffax = f.add_subplot(6, 2, 9)
    sl_lenax = f.add_subplot(6, 2, 11)

    res_rvax = f.add_subplot(6, 2, 2)
    res_loggax = f.add_subplot(6, 2, 4)
    res_mhax = f.add_subplot(6, 2, 6)
    res_alphaax = f.add_subplot(6, 2, 8)
    res_teffax = f.add_subplot(6, 2, 10)
    res_lenax = f.add_subplot(6, 2, 12)

    allax = f.get_axes()

    sl_val = sorted(sl_val)
    res_val = sorted(res_val)

    cal_star_info = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    print [x[0] for x in cal_star_info]
    if starname in [x[0] for x in cal_star_info]:
        print starname + ' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]

    order_info = scipy.genfromtxt('/u/rbentley/metallicity/reduced_orders.dat', delimiter='\t', skip_header=1,
                                  dtype=None)

    print sl_val[0][0]

    unmasked_rv = None
    if sl_val[0][0] == 0.0:
        unmasked_rv = (sl_val[0][2], sl_val[0][3], sl_val[0][4])
    else:
        unmasked_results = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_0.0_' + starname + '_order34-36.h5')
        unmasked_rv = (
        unmasked_results.median['vrad_3'], unmasked_results.median['vrad_4'], unmasked_results.median['vrad_5'])

    sl_rvax.errorbar([i[0] for i in sl_val], [i[2] - unmasked_rv[0] for i in sl_val], [i[9][5] for i in sl_val],
                     color='red', fmt='o', linestyle='--', capsize=5, label='Order 34')
    sl_rvax.errorbar([i[0] for i in sl_val], [i[3] - unmasked_rv[1] for i in sl_val], [i[9][6] for i in sl_val],
                     color='green', fmt='o', linestyle='--', capsize=5, label='Order 35')
    sl_rvax.errorbar([i[0] for i in sl_val], [i[4] - unmasked_rv[2] for i in sl_val], [i[9][7] for i in sl_val],
                     color='blue', fmt='o', linestyle='--', capsize=5, label='Order 36')

    sl_rvax.set_ylabel('Radial Velocity difference (km/s)', fontsize=8)
    sl_rvax.set_title(starname + ' $S_{\lambda}$ masked fit responses', fontsize=12)

    sl_loggax.errorbar([i[0] for i in sl_val], [i[5] - float(cal_star_info[3]) for i in sl_val],
                       [i[9][1] for i in sl_val], color='black', fmt='o', linestyle='--', capsize=5)
    sl_loggax.set_ylabel('Log g difference', fontsize=8)

    sl_mhax.errorbar([i[0] for i in sl_val], [i[6] - float(cal_star_info[1]) for i in sl_val],
                     [i[9][2] for i in sl_val], color='black', fmt='o', linestyle='--', capsize=5)
    sl_mhax.set_ylabel('[M/H] difference', fontsize=8)

    sl_alphaax.errorbar([i[0] for i in sl_val], [i[7] - float(cal_star_info[4]) for i in sl_val],
                        [i[9][3] for i in sl_val], color='black', fmt='o', linestyle='--', capsize=5)
    sl_alphaax.set_ylabel('$alpha$ difference', fontsize=8)

    sl_teffax.errorbar([i[0] for i in sl_val], [i[8] - float(cal_star_info[2]) for i in sl_val],
                       [i[9][0] for i in sl_val], color='black', fmt='o', linestyle='--', capsize=5)
    sl_teffax.set_ylabel('$T_{eff}$ (K) difference', fontsize=8)

    sl_lenax.plot([i[0] for i in sl_val], [i[1] for i in sl_val], color='black')
    sl_lenax.axhline(y=970 * 4, color='k', linestyle='--', label='Total number of data points')

    sl_lenax.set_xlabel('$S_{\lambda}$ cutoff', fontsize=8)
    sl_lenax.set_ylabel('# points masked', fontsize=8)
    # sl_lenax.legend(loc='center right', fontsize=13)

    sl_rvax.set_xscale('log')
    sl_loggax.set_xscale('log')
    sl_mhax.set_xscale('log')
    sl_alphaax.set_xscale('log')
    sl_teffax.set_xscale('log')
    sl_lenax.set_xscale('log')

    res_rvax.errorbar([i[0] for i in res_val], [i[2] - unmasked_rv[0] for i in res_val], [i[9][5] for i in res_val],
                      color='red', fmt='o', linestyle='--', capsize=5, label='Order 34')
    res_rvax.errorbar([i[0] for i in res_val], [i[3] - unmasked_rv[1] for i in res_val], [i[9][6] for i in res_val],
                      color='green', fmt='o', linestyle='--', capsize=5, label='Order 35')
    res_rvax.errorbar([i[0] for i in res_val], [i[4] - unmasked_rv[2] for i in res_val], [i[9][7] for i in res_val],
                      color='blue', fmt='o', linestyle='--', capsize=5, label='Order 36')

    res_rvax.set_ylabel('Radial Velocity difference (km/s)', fontsize=8)
    res_rvax.set_title(starname + ' residual masked fit responses', fontsize=12)

    res_loggax.errorbar([i[0] for i in res_val], [i[5] - float(cal_star_info[3]) for i in res_val],
                        [i[9][1] for i in res_val], color='black', fmt='o', linestyle='--', capsize=5)
    res_loggax.set_ylabel('Log g difference', fontsize=8)

    res_mhax.errorbar([i[0] for i in res_val], [i[6] - float(cal_star_info[1]) for i in res_val],
                      [i[9][2] for i in res_val], color='black', fmt='o', linestyle='--', capsize=5)
    res_mhax.set_ylabel('[M/H] difference', fontsize=8)

    res_alphaax.errorbar([i[0] for i in res_val], [i[7] - float(cal_star_info[4]) for i in res_val],
                         [i[9][3] for i in res_val], color='black', fmt='o', linestyle='--', capsize=5)
    res_alphaax.set_ylabel('$alpha$ difference', fontsize=8)

    res_teffax.errorbar([i[0] for i in res_val], [i[8] - float(cal_star_info[2]) for i in res_val],
                        [i[9][0] for i in res_val], color='black', fmt='o', linestyle='--', capsize=5)
    res_teffax.set_ylabel('$T_{eff}$ difference(K)', fontsize=8)

    res_lenax.plot([i[0] for i in res_val], [i[1] for i in res_val], color='black')
    res_lenax.axhline(y=970 * 4, color='k', linestyle='--', label='Total number of data points')
    res_lenax.set_xlabel('Residual cutoff', fontsize=12)
    res_lenax.set_ylabel('# points masked', fontsize=12)
    res_lenax.legend(loc='center right', fontsize=13)

    res_rvax.invert_xaxis()
    res_loggax.invert_xaxis()
    res_mhax.invert_xaxis()
    res_alphaax.invert_xaxis()
    res_teffax.invert_xaxis()
    res_lenax.invert_xaxis()

    for ax in allax:
        ax.axhline(y=0.0, color='y', linestyle='--', label='APOGEE value')

    res_rvax.legend(bbox_to_anchor=(0.95, 0.75), fontsize=10)

    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=12)
    if savefig:
        plt.savefig(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/sl-res_response_plot_' + starname + '.png')
    plt.show()


def plot_sl_res_response_allstar_three_order(sl_val_list, res_val_list, starname_list, savefig=False):
    f = plt.figure(figsize=(24, 12))

    # sl_rvax  = f.add_subplot(6,2,1)
    sl_loggax = f.add_subplot(4, 2, 1)
    sl_mhax = f.add_subplot(4, 2, 3)
    sl_alphaax = f.add_subplot(4, 2, 5)
    sl_teffax = f.add_subplot(4, 2, 7)
    # sl_lenax = f.add_subplot(6,2,11)

    # res_rvax  = f.add_subplot(6,2,2)
    res_loggax = f.add_subplot(4, 2, 2)
    res_mhax = f.add_subplot(4, 2, 4)
    res_alphaax = f.add_subplot(4, 2, 6)
    res_teffax = f.add_subplot(4, 2, 8)
    # res_lenax = f.add_subplot(6,2,12)

    allax = f.get_axes()

    color = iter(cm.rainbow(np.linspace(0, 1, len(sl_val_list))))

    for j in range(len(sl_val_list)):
        c = next(color)
        sl_val = sl_val_list[j]
        res_val = res_val_list[j]

        sl_val = sorted(sl_val)
        res_val = sorted(res_val)

        starname = starname_list[j]

        cal_star_info = list(
            scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
        print [x[0] for x in cal_star_info]
        if starname in [x[0] for x in cal_star_info]:
            print starname + ' is a calibrator star'
            star_ind = [x[0] for x in cal_star_info].index(starname)
            cal_star_info = cal_star_info[star_ind]

        order_info = scipy.genfromtxt('/u/rbentley/metallicity/reduced_orders.dat', delimiter='\t', skip_header=1,
                                      dtype=None)

        unmasked_rv = None
        if sl_val[0][0] == 0.0:
            unmasked_rv = (sl_val[0][2], sl_val[0][3], sl_val[0][4])
        else:
            unmasked_results = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_0.0_' + starname + '_order34-36.h5')
            unmasked_rv = (unmasked_results.median['vrad_3'], unmasked_results.median['vrad_4'], unmasked_results.median['vrad_5'])

        # sl_rvax.errorbar([i[0] for i in sl_val],[i[2]-unmasked_rv[0] for i in sl_val],[i[10][5] for i in sl_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
        # sl_rvax.errorbar([i[0] for i in sl_val],[i[3]-unmasked_rv[1] for i in sl_val],[i[10][6] for i in sl_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
        # sl_rvax.errorbar([i[0] for i in sl_val],[i[4]-unmasked_rv[2] for i in sl_val],[i[10][7] for i in sl_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
        # sl_rvax.errorbar([i[0] for i in sl_val],[i[5]-unmasked_rv[3] for i in sl_val],[i[10][8] for i in sl_val], fmt='o',linestyle='--',capsize=5,label='Order 37')
        # sl_rvax.set_ylabel('Radial Velocity difference (km/s)',fontsize=8)
        sl_loggax.set_title('$S_{\lambda}$ masked fit responses for all stars', fontsize=12)

        sl_loggax.errorbar([i[0] for i in sl_val], [i[5] - float(cal_star_info[3]) for i in sl_val],
                           [i[9][1] for i in sl_val], c=c, fmt='o', linestyle='--', capsize=5)
        sl_loggax.set_ylabel('Log g difference', fontsize=12)

        sl_mhax.errorbar([i[0] for i in sl_val], [i[6] - float(cal_star_info[1]) for i in sl_val],
                         [i[9][2] for i in sl_val], c=c, fmt='o', linestyle='--', capsize=5)
        sl_mhax.set_ylabel('[M/H] difference', fontsize=12)

        sl_alphaax.errorbar([i[0] for i in sl_val], [i[7] - float(cal_star_info[4]) for i in sl_val],
                            [i[9][3] for i in sl_val], c=c, fmt='o', linestyle='--', capsize=5)
        sl_alphaax.set_ylabel('$alpha$ difference', fontsize=12)

        sl_teffax.errorbar([i[0] for i in sl_val], [i[8] - float(cal_star_info[2]) for i in sl_val],
                           [i[9][0] for i in sl_val], c=c, fmt='o', linestyle='--', capsize=5)
        sl_teffax.set_ylabel('$T_{eff}$ (K) difference', fontsize=12)

        # sl_lenax.plot([i[0] for i in sl_val],[i[1] for i in sl_val], color='black')
        # sl_lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')

        # sl_lenax.set_xlabel('$S_{\lambda}$ cutoff',fontsize=8)
        # sl_lenax.set_ylabel('# points masked',fontsize=8)
        # sl_lenax.legend(loc='center right', fontsize=13)

        # sl_rvax.set_xscale('log')
        sl_loggax.set_xscale('log')
        sl_loggax.set_xlim(0, 10)
        sl_loggax.set_ylim(-2, 3.5)

        sl_mhax.set_xscale('log')
        sl_mhax.set_xlim(0, 10)
        sl_mhax.set_ylim(-1.5, 0.7)

        sl_alphaax.set_xscale('log')
        sl_alphaax.set_xlim(0, 10)
        sl_alphaax.set_ylim(-0.5, 1.)

        sl_teffax.set_xscale('log')
        sl_teffax.set_xlim(0, 10)
        sl_teffax.set_ylim(-700, 700)

        # sl_lenax.set_xscale('log')

        # res_rvax.errorbar([i[0] for i in res_val],[i[2]-unmasked_rv[0] for i in res_val],[i[10][5] for i in res_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
        # res_rvax.errorbar([i[0] for i in res_val],[i[3]-unmasked_rv[1] for i in res_val],[i[10][6] for i in res_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
        # res_rvax.errorbar([i[0] for i in res_val],[i[4]-unmasked_rv[2] for i in res_val],[i[10][7] for i in res_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
        # res_rvax.errorbar([i[0] for i in res_val],[i[5]-unmasked_rv[3] for i in res_val],[i[10][8] for i in res_val], fmt='o',linestyle='--',capsize=5,label='Order 37')

        # res_rvax.set_ylabel('Radial Velocity difference (km/s)',fontsize=8)
        res_loggax.set_title('Residual masked fit responses for all stars (PHOENIX grid, orders 34-36)', fontsize=12)

        res_loggax.errorbar([i[0] for i in res_val], [i[5] - float(cal_star_info[3]) for i in res_val],
                            [i[9][1] for i in res_val], c=c, fmt='o', linestyle='--', capsize=5, label=starname)
        # res_loggax.set_ylabel('Log g difference',fontsize=10)

        res_mhax.errorbar([i[0] for i in res_val], [i[6] - float(cal_star_info[1]) for i in res_val],
                          [i[9][2] for i in res_val], c=c, fmt='o', linestyle='--', capsize=5)
        # res_mhax.set_ylabel('[M/H] difference',fontsize=10)

        res_alphaax.errorbar([i[0] for i in res_val], [i[7] - float(cal_star_info[4]) for i in res_val],
                             [i[9][3] for i in res_val], c=c, fmt='o', linestyle='--', capsize=5)
        # res_alphaax.set_ylabel('$\alpha$ difference',fontsize=10)

        res_teffax.errorbar([i[0] for i in res_val], [i[8] - float(cal_star_info[2]) for i in res_val],
                            [i[9][0] for i in res_val], c=c, fmt='o', linestyle='--', capsize=5)
        # res_teffax.set_ylabel('$T_{eff}$ difference(K)',fontsize=10)

        # res_lenax.plot([i[0] for i in res_val],[i[1] for i in res_val], color='black')

        # res_lenax.set_xlabel('Residual cutoff',fontsize=12)
        # res_lenax.set_ylabel('# points masked',fontsize=12)

        # res_rvax.invert_xaxis()
        res_loggax.invert_xaxis()
        res_loggax.set_ylim(-2, 3.5)

        res_mhax.invert_xaxis()
        res_mhax.set_ylim(-1.5, 0.7)

        res_alphaax.invert_xaxis()
        res_alphaax.set_ylim(-0.5, 1.)

        res_teffax.invert_xaxis()
        res_teffax.set_ylim(-2000, 500)

        print "see s_l cuts here:", [i[0] for i in sl_val]
        # res_lenax.invert_xaxis()

        # if j ==0:
        # res_lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')

    for ax in allax:
        ax.axhline(y=0.0, color='y', linestyle='--', label='APOGEE value')

    # res_rvax.legend(bbox_to_anchor=(0.95,0.75), fontsize=10)

    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=12)

    sl_avg_fit, sl_avg = get_averages_sl(starname_list)

    sl_teffax.plot(sl_avg, [i[0] for i in sl_avg_fit], color = 'black')
    sl_loggax.plot(sl_avg, [i[1] for i in sl_avg_fit], color = 'black')
    sl_mhax.plot(sl_avg, [i[2] for i in sl_avg_fit], color = 'black')
    sl_alphaax.plot(sl_avg, [i[3] for i in sl_avg_fit], color = 'black')

    res_avg_fit, res_avg = get_averages_res(starname_list)

    res_teffax.plot(res_avg, [i[0] for i in res_avg_fit], color = 'black')
    res_loggax.plot(res_avg, [i[1] for i in res_avg_fit], color = 'black', label='Average')
    res_mhax.plot(res_avg, [i[2] for i in res_avg_fit], color = 'black')
    res_alphaax.plot(res_avg, [i[3] for i in res_avg_fit], color = 'black')

    res_loggax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

    sl_teffax.set_xlabel('$S_{\lambda}$ lower bound')
    res_teffax.set_xlabel('Residual cutoff')

    if savefig:
        plt.savefig(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/sl-res_response_plot_all_082819.png')
    plt.show()

def plot_sl_res_response_allstar_four_order(sl_val_list,res_val_list,starname_list, savefig=False):
    f= plt.figure(figsize=(24,12))

    
    #sl_rvax  = f.add_subplot(6,2,1)
    sl_loggax  = f.add_subplot(4,2,1)
    sl_mhax  = f.add_subplot(4,2,3)
    sl_alphaax = f.add_subplot(4,2,5)
    sl_teffax = f.add_subplot(4,2,7)
    #sl_lenax = f.add_subplot(6,2,11)


    #res_rvax  = f.add_subplot(6,2,2)
    res_loggax  = f.add_subplot(4,2,2)
    res_mhax  = f.add_subplot(4,2,4)
    res_alphaax = f.add_subplot(4,2,6)
    res_teffax = f.add_subplot(4,2,8)
    #res_lenax = f.add_subplot(6,2,12)
    
    allax = f.get_axes()

    color=iter(cm.rainbow(np.linspace(0,1,len(sl_val_list))))

    for j in range(len(sl_val_list)):
        c=next(color)
        sl_val = sl_val_list[j]
        res_val = res_val_list[j]
        
        sl_val = sorted(sl_val)
        res_val = sorted(res_val)
        
        starname = starname_list[j]

        cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))
        print [x[0] for x in cal_star_info]
        if starname in [x[0] for x in cal_star_info]:
            print starname+' is a calibrator star'
            star_ind = [x[0] for x in cal_star_info].index(starname)
            cal_star_info = cal_star_info[star_ind]

        order_info = scipy.genfromtxt('/u/rbentley/metallicity/reduced_orders.dat', delimiter='\t', skip_header=1,dtype=None)

    
        unmasked_rv = None
        if sl_val[0][0] == 0.0:
            unmasked_rv = (sl_val[0][2],sl_val[0][3],sl_val[0][4],sl_val[0][5])
        else:
            unmasked_results = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/masked_sl_cutoff_0.0_'+starname+'order34-37.h5')
            unmasked_rv = (unmasked_results.median['vrad_3'],unmasked_results.median['vrad_4'],unmasked_results.median['vrad_5'],unmasked_results.median['vrad_6'])
        
        #sl_rvax.errorbar([i[0] for i in sl_val],[i[2]-unmasked_rv[0] for i in sl_val],[i[10][5] for i in sl_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
        #sl_rvax.errorbar([i[0] for i in sl_val],[i[3]-unmasked_rv[1] for i in sl_val],[i[10][6] for i in sl_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
        #sl_rvax.errorbar([i[0] for i in sl_val],[i[4]-unmasked_rv[2] for i in sl_val],[i[10][7] for i in sl_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
        #sl_rvax.errorbar([i[0] for i in sl_val],[i[5]-unmasked_rv[3] for i in sl_val],[i[10][8] for i in sl_val], fmt='o',linestyle='--',capsize=5,label='Order 37')
        #sl_rvax.set_ylabel('Radial Velocity difference (km/s)',fontsize=8)
        sl_loggax.set_title('$S_{\lambda}$ masked fit responses for all stars',fontsize=12)

        sl_loggax.errorbar([i[0] for i in sl_val],[i[6]-float(cal_star_info[3]) for i in sl_val],[i[10][1] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_loggax.set_ylabel('Log g difference',fontsize=12)

        sl_mhax.errorbar([i[0] for i in sl_val],[i[7]-float(cal_star_info[1]) for i in sl_val],[i[10][2] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_mhax.set_ylabel('[M/H] difference',fontsize=12)

        sl_alphaax.errorbar([i[0] for i in sl_val],[i[8]-float(cal_star_info[4]) for i in sl_val],[i[10][3] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_alphaax.set_ylabel('$alpha$ difference',fontsize=12)

        sl_teffax.errorbar([i[0] for i in sl_val],[i[9]-float(cal_star_info[2]) for i in sl_val],[i[10][0] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_teffax.set_ylabel('$T_{eff}$ (K) difference',fontsize=12)

        #sl_lenax.plot([i[0] for i in sl_val],[i[1] for i in sl_val], color='black')
        #sl_lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')

        #sl_lenax.set_xlabel('$S_{\lambda}$ cutoff',fontsize=8)
        #sl_lenax.set_ylabel('# points masked',fontsize=8)
        #sl_lenax.legend(loc='center right', fontsize=13)

        #sl_rvax.set_xscale('log')
        sl_loggax.set_xscale('log')
        sl_loggax.set_xlim(0,10)
        sl_loggax.set_ylim(-2,3.5)
        
        sl_mhax.set_xscale('log')
        sl_mhax.set_xlim(0,10)
        sl_mhax.set_ylim(-1.5,0.7)
        
        sl_alphaax.set_xscale('log')
        sl_alphaax.set_xlim(0,10)
        sl_alphaax.set_ylim(-0.5,1.)
        
        sl_teffax.set_xscale('log')
        sl_teffax.set_xlim(0,10)
        sl_teffax.set_ylim(-2000,2000)

        #sl_lenax.set_xscale('log')


        #res_rvax.errorbar([i[0] for i in res_val],[i[2]-unmasked_rv[0] for i in res_val],[i[10][5] for i in res_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
        #res_rvax.errorbar([i[0] for i in res_val],[i[3]-unmasked_rv[1] for i in res_val],[i[10][6] for i in res_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
        #res_rvax.errorbar([i[0] for i in res_val],[i[4]-unmasked_rv[2] for i in res_val],[i[10][7] for i in res_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
        #res_rvax.errorbar([i[0] for i in res_val],[i[5]-unmasked_rv[3] for i in res_val],[i[10][8] for i in res_val], fmt='o',linestyle='--',capsize=5,label='Order 37')
    
        #res_rvax.set_ylabel('Radial Velocity difference (km/s)',fontsize=8)
        res_loggax.set_title('Residual masked fit responses for all stars',fontsize=12)
    
        res_loggax.errorbar([i[0] for i in res_val],[i[6]-float(cal_star_info[3]) for i in res_val],[i[10][1] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5,label=starname)
        #res_loggax.set_ylabel('Log g difference',fontsize=10)
    

        res_mhax.errorbar([i[0] for i in res_val],[i[7]-float(cal_star_info[1]) for i in res_val],[i[10][2] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5)
        #res_mhax.set_ylabel('[M/H] difference',fontsize=10)

        res_alphaax.errorbar([i[0] for i in res_val],[i[8]-float(cal_star_info[4]) for i in res_val],[i[10][3] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5)
        #res_alphaax.set_ylabel('$\alpha$ difference',fontsize=10)

        res_teffax.errorbar([i[0] for i in res_val],[i[9]-float(cal_star_info[2]) for i in res_val],[i[10][0] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5)
        #res_teffax.set_ylabel('$T_{eff}$ difference(K)',fontsize=10)

        #res_lenax.plot([i[0] for i in res_val],[i[1] for i in res_val], color='black')

        #res_lenax.set_xlabel('Residual cutoff',fontsize=12)
        #res_lenax.set_ylabel('# points masked',fontsize=12)

    
        #res_rvax.invert_xaxis()
        res_loggax.invert_xaxis()
        res_loggax.set_ylim(-2,3.5)
        
        res_mhax.invert_xaxis()
        res_mhax.set_ylim(-1.5,0.7)
        
        res_alphaax.invert_xaxis()
        res_alphaax.set_ylim(-0.5,1.)
        
        res_teffax.invert_xaxis()
        res_teffax.set_ylim(-2000,500)
        
        print "see s_l cuts here:", [i[0] for i in sl_val]
        #res_lenax.invert_xaxis()

        #if j ==0:
            #res_lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')

    for ax in allax:
        ax.axhline(y=0.0, color='y',linestyle='--',label='APOGEE value')

    #res_rvax.legend(bbox_to_anchor=(0.95,0.75), fontsize=10)

    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    plt.rc('axes',labelsize=12)

    avg_fit,sl_avg = get_averages_sl(starname_list)

    sl_teffax.plot(sl_avg,[i[0] for i in avg_fit],label = 'Average')
    sl_loggax.plot(sl_avg,[i[1] for i in avg_fit])
    sl_mhax.plot(sl_avg,[i[2] for i in avg_fit])
    sl_alphaax.plot(sl_avg,[i[3] for i in avg_fit])
    res_loggax.plot([],[],label = 'Average')

    res_loggax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

    sl_teffax.set_xlabel('$S_{\lambda}$ lower bound')
    res_teffax.set_xlabel('Residual cutoff')

    if savefig:
        plt.savefig('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/sl-res_response_plot_all_082819.png')
    plt.show()

def master_sl_plot():

    cal_stars = ['NGC6791_J19205+3748282', 'NGC6791_J19213390+3750202', \
                 'NGC6819_J19413439+4017482','M71_J19534827+1848021', \
                 'M5 J15190+0208','TYC 3544','NGC6819_J19411+4010517']

    spec='/u/ghezgroup/data/metallicity/nirspec/spectra/'

    g = load_full_grid()

    res_param_info = []
    sl_param_info = []
    
    for star in cal_stars:
        res_val, combined_data_mask_model = residual_masked_param_info(star,g,specdir=spec,snr=30.,nnorm=2)
        res_param_info += [res_val]
        sl_val, combined_data_mask_model = sl_masked_param_info(star,g,specdir=spec,snr=30.,nnorm=2)
        sl_param_info += [sl_val]

    plot_sl_res_response_allstar(sl_param_info,res_param_info,cal_stars, savefig=True)


def get_averages_sl(starname_list):
    
    avgs = []
    
    sl_vals = []
    
    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))
    
    for sl in range(10):
        teff = []
        logg = []
        mh = []
        alpha = []
        h5_files = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_'+str(float(sl))+'_*_order34-36_phoenix.h5')
        
        
        for filename in h5_files:
            for starname in starname_list:
                if starname in filename:
                    star_ind = [x[0] for x in cal_star_info_all].index(starname)
                    cal_star_info = cal_star_info_all[star_ind]
        
            gc_result = MultiNestResult.from_hdf5(filename)
            teff += [gc_result.median['teff_0']-float(cal_star_info[2])]
            logg += [gc_result.median['logg_0']-float(cal_star_info[3])]
            mh += [gc_result.median['mh_0']-float(cal_star_info[1])]
            alpha += [gc_result.median['alpha_0']-float(cal_star_info[4])]
    
        avgs += [[np.mean(teff),np.mean(logg),np.mean(mh),np.mean(alpha)]]
    return avgs, range(10)


def get_averages_res(starname_list):
    avgs = []

    sl_vals = []

    cal_star_info_all = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))

    for sl in np.arange(0.025, 0.2, 0.025):
        teff = []
        logg = []
        mh = []
        alpha = []
        h5_files = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/masked_res_cutoff_' + str(float(sl)) + '_*_order34-36_phoenix.h5')

        for filename in h5_files:
            for starname in starname_list:
                if starname in filename:
                    star_ind = [x[0] for x in cal_star_info_all].index(starname)
                    cal_star_info = cal_star_info_all[star_ind]

            gc_result = MultiNestResult.from_hdf5(filename)
            teff += [gc_result.median['teff_0'] - float(cal_star_info[2])]
            logg += [gc_result.median['logg_0'] - float(cal_star_info[3])]
            mh += [gc_result.median['mh_0'] - float(cal_star_info[1])]
            alpha += [gc_result.median['alpha_0'] - float(cal_star_info[4])]

        avgs += [[np.mean(teff), np.mean(logg), np.mean(mh), np.mean(alpha)]]
    return avgs, np.arange(0.025, 0.2, 0.025)


def get_apogee_deltas(starname):
    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    print [x[0] for x in cal_star_info]
    if starname in [x[0] for x in cal_star_info]:
        print starname + ' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]
    pho_3_order_fitpath = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_0.0_'+starname+'_order34-36.h5'
    pho_4_order_fitpath = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masked_fit_results/orders34-35-36-37/masked_sl_cutoff_0.0_'+starname+'_order34-37.h5'
    bosz_3_order_fitpath = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masked_fit_results/orders34-35-36/mh_masked_sl_cutoff_0.0_'+starname+'_order34-36_bosz.h5'
    bosz_4_order_fitpath = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masked_fit_results/orders34-35-36-37/mh_masked_sl_cutoff_0.0_'+starname+'_order34-37_bosz_order37trimmed.h5'

    pho_3_order_fit = MultiNestResult.from_hdf5(pho_3_order_fitpath)
    pho_4_order_fit = MultiNestResult.from_hdf5(pho_4_order_fitpath)
    bosz_3_order_fit = MultiNestResult.from_hdf5(bosz_3_order_fitpath)
    bosz_4_order_fit = MultiNestResult.from_hdf5(bosz_4_order_fitpath)

    print pho_3_order_fit, pho_3_order_fitpath

    print pho_4_order_fit, pho_4_order_fitpath

    print bosz_3_order_fit, bosz_3_order_fitpath

    print bosz_4_order_fit, bosz_4_order_fitpath

    f = open('/u/ghezgroup/data/metallicity/nirspec/calibrator_param_offsets.tsv','a+')
    f.write(starname+'\t'+str(pho_3_order_fit.median['teff_0']-cal_star_info[2])+'\t'+str(pho_3_order_fit.median['logg_0']-cal_star_info[3])+'\t'+str(pho_3_order_fit.median['mh_0']-cal_star_info[1])\
            +'\t' + str(pho_3_order_fit.median['alpha_0'] - cal_star_info[4]))

    f.write('\t'+str(pho_4_order_fit.median['teff_0']-cal_star_info[2])+'\t'+str(pho_4_order_fit.median['logg_0']-cal_star_info[3])+'\t'+str(pho_4_order_fit.median['mh_0']-cal_star_info[1])\
            +'\t' + str(pho_4_order_fit.median['alpha_0'] - cal_star_info[4]))

    f.write('\t'+str(bosz_3_order_fit.median['teff_0']-cal_star_info[2])+'\t'+str(bosz_3_order_fit.median['logg_0']-cal_star_info[3])+'\t'+str(bosz_3_order_fit.median['mh_0']-cal_star_info[1])\
            +'\t' + str(bosz_3_order_fit.median['alpha_0'] - cal_star_info[4]))

    f.write('\t'+str(bosz_4_order_fit.median['teff_0']-cal_star_info[2])+'\t'+str(bosz_4_order_fit.median['logg_0']-cal_star_info[3])+'\t'+str(bosz_4_order_fit.median['mh_0']-cal_star_info[1])\
            +'\t' + str(bosz_4_order_fit.median['alpha_0'] - cal_star_info[4])+'\n')

    f.close()

def get_apogee_deltas_masked(starname):
    cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    print [x[0] for x in cal_star_info]
    if starname in [x[0] for x in cal_star_info]:
        print starname + ' is a calibrator star'
        star_ind = [x[0] for x in cal_star_info].index(starname)
        cal_star_info = cal_star_info[star_ind]

    sl_fitpath = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masked_fit_results/orders34-35-36-37/masked_sl_cutoff_6.0_'+starname+'_order34-37.h5'
    res_fitpath = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/masked_fit_results/orders34-35-36-37/masked_res_cutoff_0.15_'+starname+'_order34-37.h5'

    sl_fit = MultiNestResult.from_hdf5(sl_fitpath)
    res_fit = MultiNestResult.from_hdf5(res_fitpath)

    f = open('/u/ghezgroup/data/metallicity/nirspec/calibrator_param_offsets_masking.tsv','a+')
    f.write(starname+'\t'+str(sl_fit.median['teff_0']-cal_star_info[2])+'\t'+str(sl_fit.median['logg_0']-cal_star_info[3])+'\t'+str(sl_fit.median['mh_0']-cal_star_info[1])\
            +'\t' + str(sl_fit.median['alpha_0'] - cal_star_info[4]))

    f.write('\t'+str(res_fit.median['teff_0']-cal_star_info[2])+'\t'+str(res_fit.median['logg_0']-cal_star_info[3])+'\t'+str(res_fit.median['mh_0']-cal_star_info[1])\
            +'\t' + str(res_fit.median['alpha_0'] - cal_star_info[4])+'\n')
    f.close()


def plot_apogee_deltas(xparam, yparam):
    cal_star_offsets = list(scipy.genfromtxt('/u/rbentley/metallicity/calibrator_param_offsets.tsv', delimiter='\t', skip_header=2,\
                         dtype=None))

    if yparam is 'mh':
        yind = [3,7,11,15,19,23]
        xind = 27
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = '[M/H]'

    elif yparam is 'teff':
        yind = [1,5,9,13,17,21]
        xind = 25
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = '$T_{eff}$'

    elif yparam is 'logg':
        yind = [2,6,10,14,18,22]
        xind = 26
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = 'log g'

    elif yparam is 'alpha':
        yind = [4,8,12,16,20,24]
        xind = 28
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = '[$\\alpha$/Fe]'


    if xparam is 'mh':
        xind = 27
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = '[M/H]'

    elif xparam is 'teff':
        xind = 25
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = '$T_{eff}$'

    elif xparam is 'logg':
        xind = 26
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = 'log g'

    elif xparam is 'alpha':
        xind = 28
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = '[$\\alpha$/Fe]'


    plt.plot(xvals,yvals[0], color='#FF0303',label="PHOENIX grid, using 3 orders",marker='o')

    plt.plot(xvals,yvals[1], color='#F5A182',label="PHOENIX grid, using 4 orders",marker='o')

    plt.plot(xvals, yvals[2], color='#0394FF',label="BOSZ grid, using 3 orders",marker='o')

    plt.plot(xvals, yvals[3], color='#52D1DE',label="BOSZ grid, using 4 orders",marker='o')

    #plt.axhline(y=np.mean(yvals[0]), color='#FF0303',label="PHOENIX 3 order offset average", linestyle='--')

    #plt.axhline(y=np.mean(yvals[1]), color='#F5A182',label="PHOENIX 4 order offset average", linestyle='--')

    #plt.axhline(y=np.mean(yvals[2]), color='#0394FF',label="BOSZ 3 order offset average", linestyle='--')

    #plt.axhline(y=np.mean(yvals[3]), color='#52D1DE',label="BOSZ 4 order offset average", linestyle='--')

    plt.axhline(y=0., color='k', linestyle='--')
    plt.xlabel("APOGEE "+xlabel)
    plt.ylabel("Best fit "+ylabel+" - APOGEE "+ylabel)
    plt.title("Best fit offset "+ylabel+" vs APOGEE "+xlabel+' (varying grids, fitted orders)')
    plt.legend()
    plt.show()

def plot_apogee_deltas_masked(xparam, yparam):
    cal_star_offsets = list(scipy.genfromtxt('/u/rbentley/metallicity/calibrator_param_offsets.tsv', delimiter='\t', skip_header=2,\
                         dtype=None))

    if yparam is 'mh':
        yind = [3,7,11,15,19,23]
        xind = 27
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = '[M/H]'

    elif yparam is 'teff':
        yind = [1,5,9,13,17,21]
        xind = 25
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = '$T_{eff}$'

    elif yparam is 'logg':
        yind = [2,6,10,14,18,22]
        xind = 26
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = 'log g'

    elif yparam is 'alpha':
        yind = [4,8,12,16,20,24]
        xind = 28
        yvals = []
        cal_star_offsets.sort(key= lambda x: x[xind])
        for p in yind: yvals += [[x[p] for x in cal_star_offsets]]
        xvals = [x[xind] for x in cal_star_offsets]
        ylabel = '[$\\alpha$/Fe]'


    if xparam is 'mh':
        xind = 27
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = '[M/H]'

    elif xparam is 'teff':
        xind = 25
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = '$T_{eff}$'

    elif xparam is 'logg':
        xind = 26
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = 'log g'

    elif xparam is 'alpha':
        xind = 28
        cal_star_offsets.sort(key= lambda x: x[xind])
        xvals = [x[xind] for x in cal_star_offsets]
        xlabel = '[$\\alpha$/Fe]'


    plt.plot(xvals,yvals[1], color='#F5A182',label="Unmasked PHOENIX grid",marker='o')

    plt.plot(xvals, yvals[4], color='#FD297E',label="Least sensitive regions masked",marker='o')

    plt.plot(xvals, yvals[5], color='#E6F11C',label="Highest residual regions masked",marker='o')

    #plt.axhline(y=np.mean(yvals[0]), color='#FF0303',label="PHOENIX 3 order offset average", linestyle='--')

    #plt.axhline(y=np.mean(yvals[1]), color='#F5A182',label="PHOENIX 4 order offset average", linestyle='--')

    #plt.axhline(y=np.mean(yvals[2]), color='#0394FF',label="BOSZ 3 order offset average", linestyle='--')

    #plt.axhline(y=np.mean(yvals[3]), color='#52D1DE',label="BOSZ 4 order offset average", linestyle='--')

    plt.axhline(y=0., color='k', linestyle='--')
    plt.xlabel("APOGEE "+xlabel)
    plt.ylabel("Best fit "+ylabel+" - APOGEE "+ylabel)
    plt.title("Best fit offset "+ylabel+" vs APOGEE "+xlabel+' (varying masks)')
    plt.legend()
    plt.show()


def make_offset_plot_k_band_unmasked(grids=None):
    snr = 30.

    outputpath = 'umasked_only'

    result_title = 'BOSZ-PHOENIX'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/'+outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/'+outputpath)

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[0:-1]]

    bosz_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    ap_values = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    chi2_vals = []

    if grids is not None:
        phoenix = grids[0]
        bosz = grids[1]

    else:
        phoenix = load_full_grid_phoenix()
        bosz = load_full_grid_bosz()


    for name in cal_star_names:
        star_ind = cal_star_names.index(name)
        cal_star_info = cal_star_info_all[star_ind]


        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + name + '_order34-36_bosz_adderr.h5')
        phoenix_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/PHOENIX_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + name + '_order34-36_phoenix_adderr.h5')

        bosz_bounds = bosz_result.calculate_sigmas(1)

        bosz_sigmas['teff'] += [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2)]
        bosz_sigmas['logg'] += [np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2)]
        bosz_sigmas['mh'] += [np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2)]
        bosz_sigmas['alpha'] += [np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        bosz_sigmas_one = [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)

        print bosz_sigmas_one, cal_star_info

        phoenix_bounds = phoenix_result.calculate_sigmas(1)

        phoenix_sigmas['teff'] += [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2)]
        phoenix_sigmas['logg'] += [np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2)]
        phoenix_sigmas['mh'] += [np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2)]
        phoenix_sigmas['alpha'] += [np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        phoenix_sigmas_one = [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]


        bosz_offsets['teff'] += [bosz_result.median['teff_0']-cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0']-cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0']-cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0']-cal_star_info[4]]

        phoenix_offsets['teff'] += [phoenix_result.median['teff_0']-cal_star_info[2]]
        phoenix_offsets['logg'] += [phoenix_result.median['logg_0']-cal_star_info[3]]
        phoenix_offsets['mh'] += [phoenix_result.median['mh_0']-cal_star_info[1]]
        phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0']-cal_star_info[4]]

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

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


        bmodel = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, bosz)

        pmodel = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, phoenix)

        amodel = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, bosz)

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1, bw2, bf2, bw3, bf3 = bmodel()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2+np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2+np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1)+len(bosz_res2)+len(bosz_res3))

        for a in phoenix_result.median.keys():
            setattr(pmodel, a, phoenix_result.median[a])

        pw1, pf1, pw2, pf2, pw3, pf3 = pmodel()
        phoenix_res1 = starspectrum34.flux.value - pf1
        phoenix_res2 = starspectrum35.flux.value - pf2
        phoenix_res3 = starspectrum36.flux.value - pf3

        phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 / (len(phoenix_res1) + len(phoenix_res2) + len(phoenix_res3))

        setattr(amodel, 'teff_0', cal_star_info[2])
        setattr(amodel, 'logg_0', cal_star_info[3])
        setattr(amodel, 'mh_0', cal_star_info[1])
        setattr(amodel, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = amodel()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))

        chi2_vals += [(np.round_(bosz_chi2,decimals=2),np.round_(phoenix_chi2,decimals=2),np.round_(apogee_chi2,decimals=2))]

        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]

    for a in phoenix_offsets.keys():
        phoenix_offsets[a] = np.array(phoenix_offsets[a])
        phoenix_sigmas[a] = np.array(phoenix_sigmas[a])
        bosz_offsets[a] = np.array(bosz_offsets[a])
        bosz_sigmas[a] = np.array(bosz_sigmas[a])
        ap_values[a] = np.array(ap_values[a])

    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys())+1, ncols=1,figsize=(16, 12))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

    x_axis_b = bosz_offsets['teff']
    x_axis_p = phoenix_offsets['teff']

    x_axis_b = np.array([float(i) for i in x_axis_b])
    x_axis_p = np.array([float(i) for i in x_axis_p])

    for i in range(len(ax)-1):
        ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

        ppopt, ppcov = curve_fit(linear_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

        ppopt2, ppcov2 = curve_fit(quad_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[phoenix_offsets.keys()[i]], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)

        bpopt, bpcov = curve_fit(linear_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[bosz_offsets.keys()[i]])

        bpopt2, bpcov2 = curve_fit(quad_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[bosz_offsets.keys()[i]])

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()

        ax[i].plot(np.sort(x_axis_p),linear_fit(np.sort(x_axis_p), ppopt[0], ppopt[1]),color='#FF0000',ls='--',alpha=0.6, label = 'Linear fit')
        ax[i].plot(np.sort(x_axis_b),linear_fit(np.sort(x_axis_b), bpopt[0], bpopt[1]),color='#3349FF',ls='--',alpha=0.6)

        ax[i].plot(np.sort(x_axis_p),quad_fit(np.sort(x_axis_p), ppopt2[0], ppopt2[1], ppopt2[2]),color='#FF0000',ls=':',alpha=0.6, label = 'Quadratic fit')
        ax[i].plot(np.sort(x_axis_b),quad_fit(np.sort(x_axis_b), bpopt2[0], bpopt2[1], ppopt2[2]),color='#3349FF',ls=':',alpha=0.6)

        fit_res_p = np.array(phoenix_offsets[phoenix_offsets.keys()[i]] - linear_fit(x_axis_p, ppopt[0], ppopt[1]))
        chi2_p_fit = np.sum((fit_res_p)**2 / (phoenix_sigmas[phoenix_offsets.keys()[i]]**2)) / (len(fit_res_p) - len(ppopt))


        fit_res_b = np.array(bosz_offsets[bosz_offsets.keys()[i]] - linear_fit(x_axis_b, bpopt[0], bpopt[1]))
        chi2_b_fit = np.sum((fit_res_b)**2 / bosz_sigmas[bosz_offsets.keys()[i]]**2) / (len(x_axis_b) - len(bpopt))

        fit_res_p2 = np.array(phoenix_offsets[phoenix_offsets.keys()[i]] - quad_fit(x_axis_p, ppopt2[0], ppopt2[1], ppopt2[2]))
        chi2_p_fit2 = np.sum((fit_res_p2)**2 / (phoenix_sigmas[phoenix_offsets.keys()[i]]**2)) / (len(fit_res_p2) - len(ppopt2))


        fit_res_b2 = np.array(bosz_offsets[bosz_offsets.keys()[i]] - quad_fit(x_axis_b, bpopt2[0], bpopt2[1], bpopt2[2]))
        chi2_b_fit2 = np.sum((fit_res_b2)**2 / bosz_sigmas[bosz_offsets.keys()[i]]**2) / (len(fit_res_b2) - len(bpopt2))

        ax[i].text(xmax/1.2, ymin/1.,str(bosz_offsets.keys()[i])+' PHOENIX R-value: '+str(np.round_(pearsonr(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]])[0],decimals=3))+\
                   '\nLinear reduced $\chi^2$: '+str(np.round_(chi2_p_fit,decimals=3))+'\nQuadratic reduced $\chi^2$: '+str(np.round_(chi2_p_fit2,decimals=3))+\
                   '\nf-test ratio:'+str(np.round_(chi2_p_fit/chi2_p_fit2,decimals=3))+'\nBOSZ R-value: '+\
                   str(np.round_(pearsonr(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]])[0],decimals=3))+\
                   '\nLinear reduced $\chi^2$: '+str(np.round_(chi2_b_fit,decimals=3))+'\nQuadratic reduced $\chi^2$: '+\
                   str(np.round_(chi2_b_fit2,decimals=3))+'\nf-test ratio:'+str(np.round_(chi2_b_fit/chi2_b_fit2,decimals=3)),fontsize=11, bbox=props)

        #ax[i].errorbar(ap_values['mh'], two_offsets[two_offsets.keys()[i]], yerr=two_sigmas[two_offsets.keys()[i]], color='#33AFFF',marker='d',label='BOSZ grid, masked at S_l = 6.0',ls='none')

        connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[phoenix_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])

        ax[i].axhline(y=0., color='k', linestyle='--')


    ax[-1].plot(x_axis_b,[x[0] for x in chi2_vals], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)
    ax[-1].plot(x_axis_p,[x[1] for x in chi2_vals],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

    connect_points(ax[-1], x_axis_b, x_axis_p, [x[0] for x in chi2_vals],
                   [x[1] for x in chi2_vals])

    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax)-1].set_xlabel('$T_{eff}$ offset ', fontsize=16)

    ax[0].set_ylabel('log g offset', fontsize=16)
    ax[1].set_ylabel(r'$\alpha$'+' offset', fontsize=16)
    ax[2].set_ylabel('$T_{eff}$ offset', fontsize=16)
    ax[3].set_ylabel('[M/H] offset', fontsize=16)

    ax[0].legend(fontsize=12)
    ax[0].set_title('Unmasked offsets for all stars vs $T_{eff}$ offset')
    plt.show()
    #plt.savefig('/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + outputpath + '_offsets_delta_teff.png')


def linear_fit(x,m,b):
    return m*x + b

def quad_fit(x,a,b,c):
    return a*x**2 + b*x + c

def connect_points(ax,x1,x2,y1,y2):
    for i in range(len(x1)):
        ax.plot([x1[i],x2[i]],[y1[i],y2[i]], color='k', alpha=0.4,linestyle='--')


def make_fit_result_plots_k_band_three_order(fitpath, new_r=None):

    snr = 30.

    if 'bosz' in fitpath:
        g = load_full_grid_bosz()
    else:
        g = load_full_grid_phoenix()

    outputpath = fitpath.split('/')[-1]

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[0:-1]]

    files = []
    for name in cal_star_names:
        files += [fitpath.replace('*',name)]

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/'+outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/'+outputpath)

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[0:-1]]

    offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    ap_values = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    chi2_vals = []

    for filepath in files:

        if '2MJ18113-30441' in filepath:
            continue

        result_title = filepath.split('/')[-1]
        result_title.replace('.h5', '')
        print filepath
        for name in cal_star_names:
            if name in filepath:
                starname = name
                star_ind = cal_star_names.index(starname)
                cal_star_info = cal_star_info_all[star_ind]
                print starname, cal_star_info


        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + starname + '_order34-36_bosz_adderr.h5')
        phoenix_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/PHOENIX_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + starname + '_order34-36_phoenix_adderr.h5')

        bosz_bounds = bosz_result.calculate_sigmas(1)

        bosz_sigmas['teff'] += [(bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2]
        bosz_sigmas['logg'] += [(bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2]
        bosz_sigmas['mh'] += [(bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2]
        bosz_sigmas['alpha'] += [(bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2]

        phoenix_bounds = phoenix_result.calculate_sigmas(1)

        phoenix_sigmas['teff'] += [(phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2]
        phoenix_sigmas['logg'] += [(phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2]
        phoenix_sigmas['mh'] += [(phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2]
        phoenix_sigmas['alpha'] += [(phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2]


        bosz_offsets['teff'] += [bosz_result.median['teff_0']-cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0']-cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0']-cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0']-cal_star_info[4]]

        phoenix_offsets['teff'] += [phoenix_result.median['teff_0']-cal_star_info[2]]
        phoenix_offsets['logg'] += [phoenix_result.median['logg_0']-cal_star_info[3]]
        phoenix_offsets['mh'] += [phoenix_result.median['mh_0']-cal_star_info[1]]
        phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0']-cal_star_info[4]]

        result = MultiNestResult.from_hdf5(filepath)

        bounds = result.calculate_sigmas(1)

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + starname + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + starname + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + starname + '_order36*.dat')

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

        if new_r is not None:

            del_w1 = starspectrum34.wavelength.value[0] / (
                        starspectrum34.wavelength.value[1] - starspectrum34.wavelength.value[0])
            sigma1 = del_w1 / new_r / (2 * np.sqrt(2 * np.log(2)))
            conv_f1 = nd.gaussian_filter1d(starspectrum34.flux.value, sigma1)
            starspectrum34 = Spectrum1D.from_array(dispersion=starspectrum34.wavelength.value, flux=conv_f1,
                                                dispersion_unit=u.angstrom,
                                                uncertainty=starspectrum34.uncertainty.value)

            del_w2 = starspectrum35.wavelength.value[0] / (
                        starspectrum35.wavelength.value[1] - starspectrum35.wavelength.value[0])
            sigma2 = del_w2 / new_r / (2 * np.sqrt(2 * np.log(2)))
            conv_f2 = nd.gaussian_filter1d(starspectrum35.flux.value, sigma2)
            starspectrum35 = Spectrum1D.from_array(dispersion=starspectrum35.wavelength.value, flux=conv_f2,
                                                dispersion_unit=u.angstrom,
                                                uncertainty=starspectrum35.uncertainty.value)

            del_w3 = starspectrum36.wavelength.value[0] / (
                        starspectrum36.wavelength.value[1] - starspectrum36.wavelength.value[0])
            sigma3 = del_w3 / new_r / (2 * np.sqrt(2 * np.log(2)))
            conv_f3 = nd.gaussian_filter1d(starspectrum36.flux.value, sigma3)
            starspectrum36 = Spectrum1D.from_array(dispersion=starspectrum36.wavelength.value, flux=conv_f3,
                                                dispersion_unit=u.angstrom,
                                                uncertainty=starspectrum36.uncertainty.value)

            model = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, g, convolve=new_r)

        else:
            model = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, g)


        for a in bosz_result.median.keys():
            setattr(model, a, bosz_result.median[a])
        bw1, bf1, bw2, bf2, bw3, bf3 = model()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2+np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2+np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1)+len(bosz_res2)+len(bosz_res3))

        for a in phoenix_result.median.keys():
            setattr(model, a, phoenix_result.median[a])
        pw1, pf1, pw2, pf2, pw3, pf3 = model()
        phoenix_res1 = starspectrum34.flux.value - pf1
        phoenix_res2 = starspectrum35.flux.value - pf2
        phoenix_res3 = starspectrum36.flux.value - pf3

        phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 / (len(phoenix_res1) + len(phoenix_res2) + len(phoenix_res3))

        for a in result.median.keys():
            setattr(model, a, result.median[a])

        w1, f1, w2, f2, w3, f3 = model()

        deltas = [result.median['teff_0'] - cal_star_info[2], result.median['logg_0'] - cal_star_info[3],
                  result.median['mh_0'] - cal_star_info[1], result.median['alpha_0'] - cal_star_info[4]]

        deltas = np.around(deltas,decimals=2)

        offsets['teff'] += [deltas[0]]
        offsets['logg'] += [deltas[1]]
        offsets['mh'] += [deltas[2]]
        offsets['alpha'] += [deltas[3]]

        sigmas['teff'] += [(bounds['teff_0'][1]-bounds['teff_0'][0])/2]
        sigmas['logg'] += [(bounds['logg_0'][1]-bounds['logg_0'][0])/2]
        sigmas['mh'] += [(bounds['mh_0'][1]-bounds['mh_0'][0])/2]
        sigmas['alpha'] += [(bounds['alpha_0'][1]-bounds['alpha_0'][0])/2]

        sigmas_one = [np.sqrt(((bounds['teff_0'][1]-bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((bounds['logg_0'][1]-bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((bounds['mh_0'][1]-bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((bounds['alpha_0'][1]-bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        sigmas_one = np.around(sigmas_one, decimals=2)

        res1 = starspectrum34.flux.value - f1
        res2 = starspectrum35.flux.value - f2
        res3 = starspectrum36.flux.value - f3

        chi2 = np.sum((res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        chi2 = chi2+np.sum((res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        chi2 = chi2+np.sum((res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        chi2 = chi2 / (len(res1)+len(res2)+len(res3))

        setattr(model, 'teff_0', cal_star_info[2])
        setattr(model, 'logg_0', cal_star_info[3])
        setattr(model, 'mh_0', cal_star_info[1])
        setattr(model, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = model()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))



        chi2_vals += [(chi2,bosz_chi2,phoenix_chi2,apogee_chi2)]

        result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1'])

        plt.savefig('/u/rbentley/localcompute/fitting_plots/'+outputpath+'/'+result_title+'_corner.pdf')
        plt.clf()

        plt.figure(figsize=(16, 12))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

        plt.text(22090, 0.7,
                 'APOGEE fitted parameters:\n$T_{eff}:$' + str(cal_star_info[2]) +'$\pm$'+ str(cal_star_info[7]) + '\n$log g:$' + str(cal_star_info[3]) +'$\pm$'+ str(cal_star_info[8]) + \
                 '\n$[M/H]:$' + str(cal_star_info[1]) +'$\pm$'+ str(cal_star_info[6]) + '\n'+r'$\alpha$:' + str(cal_star_info[4]) +'$\pm$'+ str(cal_star_info[9]) + '\n$\chi^2$ at APOGEE values:' + str(np.round_(apogee_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.text(22090, 0.45,
                 'Fit offsets:\n$\Delta T_{eff}:$' + str(deltas[0]) +'$\pm$'+ str(sigmas_one[0]) + '\n$\Delta log g:$' + str(deltas[1]) +'$\pm$'+ str(sigmas_one[1]) + \
                 '\n$\Delta [M/H]:$' + str(deltas[2]) +'$\pm$'+ str(sigmas_one[2]) + '\n'+r'$\Delta$$\alpha$:' + str(deltas[3]) +'$\pm$'+ str(sigmas_one[3]) + '\n$\chi^2$:' + str(np.round_(chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.plot(starspectrum34.wavelength.value / (bosz_result.median['vrad_3'] / 3e5 + 1.0), starspectrum34.flux.value,
                 color='#000000', label='Data',linewidth=5.0)
        plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0), starspectrum35.flux.value,
                 color='#000000',linewidth=5.0)
        plt.plot(starspectrum36.wavelength.value / (bosz_result.median['vrad_5'] / 3e5 + 1.0), starspectrum36.flux.value,
                 color='#000000',linewidth=5.0)

        plt.plot(w2 / (result.median['vrad_4'] / 3e5 + 1.0), f2, color='#33AFFF', label='Model/Residuals',linewidth=5.0)

        plt.plot(w2 / (result.median['vrad_4'] / 3e5 + 1.0), res2, color='#33AFFF',linewidth=5.0)

        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
        plt.axhline(y=-0.05, color='k', linestyle='--')

        plt.xlim(21900,22100)
        plt.ylim(-0.2,1.3)

        plt.legend(loc='center left', fontsize=16)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        plt.title('R=5000 fits and residuals for '+starname)
        plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=15)
        plt.savefig('/u/rbentley/localcompute/fitting_plots/'+outputpath+'/'+result_title+'_spectrum.pdf')
        plt.clf()

        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]

    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys())+1, ncols=1,figsize=(16, 12))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

    x_axis_n = ap_values['mh']
    x_axis_b = ap_values['mh']
    x_axis_p = ap_values['mh']

    x_axis_n = np.array([float(i) for i in x_axis_n])
    x_axis_b = np.array([float(i) for i in x_axis_b])
    x_axis_p = np.array([float(i) for i in x_axis_p])

    for i in range(len(ax)-1):

        ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

        #ppopt, ppcov = curve_fit(linear_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

        #ppopt2, ppcov2 = curve_fit(quad_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[phoenix_offsets.keys()[i]], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)

        #bpopt, bpcov = curve_fit(linear_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[bosz_offsets.keys()[i]])

        #bpopt2, bpcov2 = curve_fit(quad_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[bosz_offsets.keys()[i]])

        ax[i].errorbar(x_axis_n, offsets[offsets.keys()[i]], yerr=sigmas[offsets.keys()[i]],color='#5cd85e',marker='d',label=outputpath,ls='none', markersize=10)

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()
        '''
        #ax[i].plot(np.sort(x_axis_p),linear_fit(np.sort(x_axis_p), ppopt[0], ppopt[1]),color='#FF0000',ls='--',alpha=0.8)
        #ax[i].plot(np.sort(x_axis_b),linear_fit(np.sort(x_axis_b), bpopt[0], bpopt[1]),color='#3349FF',ls='--',alpha=0.8)

        #ax[i].plot(np.sort(x_axis_p),quad_fit(np.sort(x_axis_p), ppopt2[0], ppopt2[1], ppopt2[2]),color='#FF0000',ls=':',alpha=0.8)
        #ax[i].plot(np.sort(x_axis_b),quad_fit(np.sort(x_axis_b), bpopt2[0], bpopt2[1], ppopt2[2]),color='#3349FF',ls=':',alpha=0.8)

        #chi2_p_fit = np.sum((phoenix_offsets[phoenix_offsets.keys()[i]] - linear_fit(x_axis_p, ppopt[0], ppopt[1]) /
                             phoenix_sigmas[phoenix_offsets.keys()[i]]) ** 2) / (len(x_axis_p) - len(ppopt))

        #chi2_b_fit = np.sum((bosz_offsets[bosz_offsets.keys()[i]] - linear_fit(x_axis_b, bpopt[0], bpopt[1]) /
                             bosz_sigmas[bosz_offsets.keys()[i]]) ** 2) / (len(x_axis_b) - len(bpopt))

        #chi2_p_fit2 = np.sum((phoenix_offsets[phoenix_offsets.keys()[i]] - quad_fit(x_axis_p, ppopt2[0], ppopt2[1], ppopt2[2]) /
                             phoenix_sigmas[phoenix_offsets.keys()[i]]) ** 2) / (len(x_axis_p) - len(ppopt2))

        #chi2_b_fit2 = np.sum((bosz_offsets[bosz_offsets.keys()[i]] - quad_fit(x_axis_b, bpopt2[0], bpopt2[1], bpopt2[2]) /
                             bosz_sigmas[bosz_offsets.keys()[i]]) ** 2) / (len(x_axis_b) - len(bpopt2))

        #ax[i].text(xmax/1.2, ymin/1.3,'PHOENIX R-value: '+str(np.round_(pearsonr(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]])[0],decimals=3))+\
                   '\nLinear reduced $\chi^2$: '+str(chi2_p_fit)+'\nQuadratic reduced $\chi^2$: '+str(chi2_p_fit2)+'\nL/Q $\chi^2$ ratio:'+str(chi2_p_fit/chi2_p_fit2)+'\nBOSZ R-value: '+\
                   str(np.round_(pearsonr(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]])[0],decimals=3))+\
                   '\nLinear reduced $\chi^2$: '+str(chi2_b_fit)+'\nQuadratic reduced $\chi^2$: '+str(chi2_b_fit2)+'\nL/Q $\chi^2$ ratio:'+str(chi2_b_fit/chi2_b_fit2),fontsize=12, bbox=props)

        #ax[i].errorbar(ap_values['mh'], two_offsets[two_offsets.keys()[i]], yerr=two_sigmas[two_offsets.keys()[i]], color='#33AFFF',marker='d',label='BOSZ grid, masked at S_l = 6.0',ls='none')
        '''
        connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[phoenix_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])

        connect_points(ax[i],x_axis_b,x_axis_n, bosz_offsets[phoenix_offsets.keys()[i]],offsets[offsets.keys()[i]])

        ax[i].axhline(y=0., color='k', linestyle='--')

    ax[-1].plot(x_axis_n,[x[0] for x in chi2_vals],color='#5cd85e',marker='d',label=outputpath,ls='none', markersize=10)
    ax[-1].plot(x_axis_b,[x[1] for x in chi2_vals], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)
    ax[-1].plot(x_axis_p,[x[2] for x in chi2_vals],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)


    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax)-1].set_xlabel('[M/H] ', fontsize=16)

    ax[0].set_ylabel('log g offset', fontsize=16)
    ax[1].set_ylabel(r'$\alpha$'+' offset', fontsize=16)
    ax[2].set_ylabel('$T_{eff}$ offset', fontsize=16)
    ax[3].set_ylabel('[M/H] offset', fontsize=16)

    ax[0].legend(fontsize=8.5)
    ax[0].set_title(outputpath+' offsets for all stars')
    plt.savefig('/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + outputpath + '_offsets.pdf')

def make_fit_result_plots_k_band_three_order_unmasked_only():

    snr = 30.

    outputpath = 'umasked_only'

    result_title = 'BOSZ-PHOENIX'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/'+outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/'+outputpath)

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[0:-1]]

    bosz_vals = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    bosz_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_vals = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    ap_values = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    chi2_vals = []

    phoenix = load_full_grid_phoenix()

    bosz = load_full_grid_bosz()


    for name in cal_star_names:
        star_ind = cal_star_names.index(name)
        cal_star_info = cal_star_info_all[star_ind]


        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + name + '_order34-36_bosz_adderr.h5')
        phoenix_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/PHOENIX_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + name + '_order34-36_phoenix_adderr.h5')

        bosz_bounds = bosz_result.calculate_sigmas(1)

        bosz_sigmas['teff'] += [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2)]
        bosz_sigmas['logg'] += [np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2)]
        bosz_sigmas['mh'] += [np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2)]
        bosz_sigmas['alpha'] += [np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        bosz_sigmas_one = [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)

        print bosz_sigmas_one, cal_star_info

        phoenix_bounds = phoenix_result.calculate_sigmas(1)

        phoenix_sigmas['teff'] += [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2)]
        phoenix_sigmas['logg'] += [np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2)]
        phoenix_sigmas['mh'] += [np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2)]
        phoenix_sigmas['alpha'] += [np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        phoenix_sigmas_one = [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)

        bosz_vals['teff'] += [bosz_result.median['teff_0']]
        bosz_vals['logg'] += [bosz_result.median['logg_0']]
        bosz_vals['mh'] += [bosz_result.median['mh_0']]
        bosz_vals['alpha'] += [bosz_result.median['alpha_0']]

        bosz_offsets['teff'] += [bosz_result.median['teff_0']-cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0']-cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0']-cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0']-cal_star_info[4]]

        phoenix_vals['teff'] += [phoenix_result.median['teff_0']]
        phoenix_vals['logg'] += [phoenix_result.median['logg_0']]
        phoenix_vals['mh'] += [phoenix_result.median['mh_0']]
        phoenix_vals['alpha'] += [phoenix_result.median['alpha_0']]

        phoenix_offsets['teff'] += [phoenix_result.median['teff_0']-cal_star_info[2]]
        phoenix_offsets['logg'] += [phoenix_result.median['logg_0']-cal_star_info[3]]
        phoenix_offsets['mh'] += [phoenix_result.median['mh_0']-cal_star_info[1]]
        phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0']-cal_star_info[4]]

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

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


        bmodel = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, bosz)

        pmodel = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, phoenix)

        amodel = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, bosz)

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1, bw2, bf2, bw3, bf3 = bmodel()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2+np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2+np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1)+len(bosz_res2)+len(bosz_res3))

        for a in phoenix_result.median.keys():
            setattr(pmodel, a, phoenix_result.median[a])

        pw1, pf1, pw2, pf2, pw3, pf3 = pmodel()
        phoenix_res1 = starspectrum34.flux.value - pf1
        phoenix_res2 = starspectrum35.flux.value - pf2
        phoenix_res3 = starspectrum36.flux.value - pf3

        phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        phoenix_chi2 = phoenix_chi2 / (len(phoenix_res1) + len(phoenix_res2) + len(phoenix_res3))

        phoenix_deltas = np.around([phoenix_result.median['teff_0'] - cal_star_info[2], phoenix_result.median['logg_0'] - cal_star_info[3],
                  phoenix_result.median['mh_0'] - cal_star_info[1], phoenix_result.median['alpha_0'] - cal_star_info[4]],decimals=2)

        bosz_deltas = np.around([bosz_result.median['teff_0'] - cal_star_info[2], bosz_result.median['logg_0'] - cal_star_info[3],
                  bosz_result.median['mh_0'] - cal_star_info[1], bosz_result.median['alpha_0'] - cal_star_info[4]],decimals=2)

        setattr(amodel, 'teff_0', cal_star_info[2])
        setattr(amodel, 'logg_0', cal_star_info[3])
        setattr(amodel, 'mh_0', cal_star_info[1])
        setattr(amodel, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = amodel()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))

        chi2_vals += [(np.round_(bosz_chi2,decimals=2),np.round_(phoenix_chi2,decimals=2),np.round_(apogee_chi2,decimals=2))]

        plt.figure(figsize=(16, 12))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

        plt.text(22090, 0.7,
                 'APOGEE fitted parameters:\n$T_{eff}:$' + str(cal_star_info[2]) +'$\pm$'+ str(cal_star_info[7]) + '\n$log g:$' + str(cal_star_info[3]) +'$\pm$'+ str(cal_star_info[8]) + \
                 '\n$[M/H]:$' + str(cal_star_info[1]) +'$\pm$'+ str(cal_star_info[6]) + '\n'+r'$\alpha$:' + str(cal_star_info[4]) +'$\pm$'+ str(cal_star_info[9]) + '\n$\chi^2$ at APOGEE values:' + str(np.round_(apogee_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.text(22090, 0.45,
                 'BOSZ fit offsets:\n$\Delta T_{eff}:$' + str(bosz_deltas[0]) +'$\pm$'+ str(bosz_sigmas_one[0]) + '\n$\Delta log g:$' + str(bosz_deltas[1]) +'$\pm$'+ str(bosz_sigmas_one[1]) + \
                 '\n$\Delta [M/H]:$' + str(bosz_deltas[2]) +'$\pm$'+ str(bosz_sigmas_one[2]) + '\n'+r'$\Delta$$\alpha$:' + str(bosz_deltas[3]) +'$\pm$'+ str(bosz_sigmas_one[3]) + '\n$\chi^2$:' + str(np.round_(bosz_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.text(22090, 0.2,
                 'PHOENIX fit offsets:\n$\Delta T_{eff}:$' + str(phoenix_deltas[0]) +'$\pm$'+ str(phoenix_sigmas_one[0]) + '\n$\Delta log g:$' + str(phoenix_deltas[1]) +'$\pm$'+ str(phoenix_sigmas_one[1]) + \
                 '\n$\Delta [M/H]:$' + str(phoenix_deltas[2]) +'$\pm$'+ str(phoenix_sigmas_one[2]) + '\n'+r'$\Delta$$\alpha$:' + str(phoenix_deltas[3]) +'$\pm$'+ str(phoenix_sigmas_one[3]) + '\n$\chi^2$:' + str(np.round_(phoenix_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.plot(starspectrum34.wavelength.value / (bosz_result.median['vrad_3'] / 3e5 + 1.0), starspectrum34.flux.value,
                 color='#000000', label='Data',linewidth=5.0)
        plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0), starspectrum35.flux.value,
                 color='#000000',linewidth=5.0)
        plt.plot(starspectrum36.wavelength.value / (bosz_result.median['vrad_5'] / 3e5 + 1.0), starspectrum36.flux.value,
                 color='#000000',linewidth=5.0)

        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bf2, color='#33AFFF', label='BOSZ Model/Residuals',linewidth=5.0)

        plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), pf2, color='#FEBE4E', label='PHOENIX Model/Residuals',linewidth=5.0)

        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bosz_res2, color='#33AFFF',linewidth=5.0)

        plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), phoenix_res2, color='#FEBE4E',linewidth=5.0)

        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
        plt.axhline(y=-0.05, color='k', linestyle='--')

        plt.xlim(21900,22100)
        plt.ylim(-0.2,1.3)

        plt.legend(loc='center left', fontsize=16)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        plt.title(result_title+' fits and residuals for '+name)
        plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=15)
        plt.savefig('/u/rbentley/localcompute/fitting_plots/'+outputpath+'/'+name+'_'+result_title+'_spectrum.pdf')
        plt.clf()

        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]

    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys())+1, ncols=1,figsize=(16, 12))

    x_axis_b = bosz_vals['mh']
    x_axis_p = phoenix_vals['mh']

    for i in range(len(ax)-1):
        ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

        #ax[i].text(200., -np.amax(phoenix_offsets[phoenix_offsets.keys()[i]])/1.3,'PHOENIX R-value: '+str(np.round_(pearsonr(phoenix_offsets['teff'], phoenix_offsets[phoenix_offsets.keys()[i]])[0],decimals=3)),fontsize=12, bbox=props)

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[phoenix_offsets.keys()[i]], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)

        #ax[i].text(200., 0.0,'BOSZ R-value: '+str(np.round_(pearsonr(bosz_offsets['teff'], bosz_offsets[bosz_offsets.keys()[i]])[0],decimals=3)),fontsize=12, bbox=props)

        #ax[i].errorbar(ap_values['mh'], two_offsets[two_offsets.keys()[i]], yerr=two_sigmas[two_offsets.keys()[i]], color='#33AFFF',marker='d',label='BOSZ grid, masked at S_l = 6.0',ls='none')

        connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[phoenix_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])

        ax[i].axhline(y=0., color='k', linestyle='--')


    ax[-1].plot(x_axis_b,[x[0] for x in chi2_vals], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)
    ax[-1].plot(x_axis_p,[x[1] for x in chi2_vals],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

    connect_points(ax[-1], x_axis_b, x_axis_p, [x[0] for x in chi2_vals],
                   [x[1] for x in chi2_vals])

    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax)-1].set_xlabel('Best-fit [M/H]', fontsize=16)

    ax[0].set_ylabel('log g offset', fontsize=16)
    ax[1].set_ylabel(r'$\alpha$'+' offset', fontsize=16)
    ax[2].set_ylabel('$T_{eff}$ offset', fontsize=16)
    ax[3].set_ylabel('[M/H] offset', fontsize=16)

    ax[0].legend(fontsize=12)
    ax[0].set_title('Unmasked offsets for all stars vs best-fit [M/H]')

    plt.savefig('/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + outputpath + '_offsets_fit_mh.png')

def make_model_three_order(spectrum1,spectrum2,spectrum3,grid, convolve=None):

    if convolve is not None:
        r_val = convolve
    else:
        r_val = 24000

    interp1 = Interpolate(spectrum1)
    convolve1 = InstrumentConvolveGrating.from_grid(grid, R=r_val)
    rot1 = RotationalBroadening.from_grid(grid, vrot=np.array([10.0]))
    norm1 = Normalize(spectrum1, 2)

    interp2 = Interpolate(spectrum2)
    convolve2 = InstrumentConvolveGrating.from_grid(grid, R=r_val)
    norm2 = Normalize(spectrum2, 2)

    interp3 = Interpolate(spectrum3)
    convolve3 = InstrumentConvolveGrating.from_grid(grid, R=r_val)
    norm3 = Normalize(spectrum3, 2)


    model = grid | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
                 convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
                 norm1 & norm2 & norm3

    return model


def find_line_residuals(fitdir, fitname, g):

    snr = 30.

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[0:-1]]

    for name in cal_star_names:
        if name in fitname:
            starname = name
            star_ind = cal_star_names.index(starname)
            cal_star_info = cal_star_info_all[star_ind]
            print starname, cal_star_info

    result = MultiNestResult.from_hdf5(fitdir+fitname)

    file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + starname + '_order34*.dat')
    file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + starname + '_order35*.dat')
    file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + starname + '_order36*.dat')

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


    model = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, g)

    for a in result.median.keys():
        setattr(model, a, result.median[a])

    w1, f1, w2, f2, w3, f3 = model()

    res1 = starspectrum34.flux.value - f1
    res2 = starspectrum35.flux.value - f2
    res3 = starspectrum36.flux.value - f3

    data_contin_coeffs34 = np.polyfit(starspectrum34.wavelength.value,starspectrum34.flux.value, 2)
    data_contin_flux34 = data_contin_coeffs34[0]*starspectrum34.wavelength.value**2+data_contin_coeffs34[1]*starspectrum34.wavelength.value+data_contin_coeffs34[2]
    data_contin34 = Spectrum1D.from_array(dispersion=starspectrum34.wavelength.value, flux=data_contin_flux34+0.04, dispersion_unit=u.angstrom, uncertainty=starspectrum34.uncertainty.value)

    data_contin_coeffs35 = np.polyfit(starspectrum35.wavelength.value,starspectrum35.flux.value, 2)
    data_contin_flux35 = data_contin_coeffs35[0]*starspectrum35.wavelength.value**2+data_contin_coeffs35[1]*starspectrum35.wavelength.value+data_contin_coeffs35[2]
    data_contin35 = Spectrum1D.from_array(dispersion=starspectrum35.wavelength.value, flux=data_contin_flux35+0.04, dispersion_unit=u.angstrom, uncertainty=starspectrum35.uncertainty.value)

    data_contin_coeffs36 = np.polyfit(starspectrum36.wavelength.value,starspectrum36.flux.value, 2)
    data_contin_flux36 = data_contin_coeffs36[0]*starspectrum36.wavelength.value**2+data_contin_coeffs36[1]*starspectrum36.wavelength.value+data_contin_coeffs36[2]
    data_contin36 = Spectrum1D.from_array(dispersion=starspectrum36.wavelength.value, flux=data_contin_flux36+0.04, dispersion_unit=u.angstrom, uncertainty=starspectrum36.uncertainty.value)

    line_flux34 = data_contin34.flux.value - f1
    line_flux35 = data_contin35.flux.value - f2
    line_flux36 = data_contin36.flux.value - f3

    w1 = w1 / (result.median['vrad_3'] / 3e5 + 1.0)
    w2 = w2 / (result.median['vrad_4'] / 3e5 + 1.0)
    w3 = w3 / (result.median['vrad_5'] / 3e5 + 1.0)

    plt.plot(w1, data_contin34.flux.value, color = 'b')
    plt.plot(w2, data_contin35.flux.value, color = 'b')
    plt.plot(w3, data_contin36.flux.value, color = 'b')

    plt.plot(w1, f1, color = 'b')
    plt.plot(w2, f2, color = 'b')
    plt.plot(w3, f3, color = 'b')

    plt.plot(w1, line_flux34, color = 'r')
    plt.plot(w2, line_flux35, color = 'r')
    plt.plot(w3, line_flux36, color = 'r')

    plt.show()

    line_w1, line_names1 = plotlines.extract_lines(arcturus=True, wave_range=[np.amin(w1),np.amax(w1)])
    line_res1 = []
    for value in line_w1:
        lname = line_names1[np.where(line_w1==value)]
        idx = np.searchsorted(w1, value, side="left")
        if idx > 0 and (idx == len(w1) or math.fabs(value - w1[idx - 1]) < math.fabs(value - w1[idx])):
            lres = res1[idx-1]
            lw = w1[idx-1]
            line_res1 += [(lname[0],lres,lw)]
        else:
            lres = res1[idx]
            lw = w1[idx]
            line_res1 += [(lname[0],lres,lw)]

    ew_bounds1 = [(22492,22498,'Fe'),(22671,22673.5,'Si')]
    eqw_vals1 = []

    for line in ew_bounds1:
        sample_points = np.where((w1 >= line[0]) & (w1 <= line[1]))
        print sample_points, w1[sample_points]
        eqw = np.trapz(line_flux34[sample_points], w1[sample_points])
        eqw = eqw/np.median(data_contin34.flux.value[sample_points])
        eqw_vals1 += [(eqw, line[2], np.mean(line[0:2]))]

    print eqw_vals1


    line_w2, line_names2 = plotlines.extract_lines(arcturus=True, wave_range=[np.amin(w2),np.amax(w2),])
    line_res2 = []
    for value in line_w2:
        lname = line_names2[np.where(line_w2==value)]
        idx = np.searchsorted(w2, value, side="left")
        if idx > 0 and (idx == len(w2) or math.fabs(value - w2[idx - 1]) < math.fabs(value - w2[idx])):
            lres = res2[idx-1]
            lw = w2[idx-1]
            line_res2 += [(lname[0],lres,lw)]
        else:
            lres = res2[idx]
            lw = w2[idx]
            line_res2 += [(lname[0],lres,lw)]

    ew_bounds2 = [(22088,22091.5,'Na'),(21921,21924,'Fe'),(21862.5,21866,'Fe')]

    eqw_vals2 = []

    for line in ew_bounds2:
        sample_points = np.where((w2 >= line[0]) & (w2 <= line[1]))
        eqw = np.trapz(line_flux35[sample_points], w2[sample_points])
        eqw = eqw/np.median(data_contin35.flux.value[sample_points])
        eqw_vals2 += [(eqw, line[2], np.mean(line[0:2]))]

    print eqw_vals2

    line_w3, line_names3 = plotlines.extract_lines(arcturus=True, wave_range=[np.amin(w3),np.amax(w3)])
    line_res3 = []
    for value in line_w3:
        lname = line_names3[np.where(line_w3==value)]
        idx = np.searchsorted(w3, value, side="left")
        if idx > 0 and (idx == len(w3) or math.fabs(value - w3[idx - 1]) < math.fabs(value - w3[idx])):
            lres = res3[idx-1]
            lw = w3[idx-1]
            line_res3 += [(lname[0],lres,lw)]
        else:
            lres = res3[idx]
            lw = w3[idx]
            line_res3 += [(lname[0],lres,lw)]

    ew_bounds3 = [(21464,21467,'Mg'),(21457,21460,'Na'),(21377,21380,'Si'),(21288,21291,'Fe'),(21218,21221.5,'Mg')]

    eqw_vals3 = []

    for line in ew_bounds3:
        sample_points = np.where((w3 >= line[0]) & (w3 <= line[1]))
        eqw = np.trapz(line_flux36[sample_points], w3[sample_points])
        eqw = eqw/np.median(data_contin36.flux.value[sample_points])
        eqw_vals3 += [(eqw, line[2], np.mean(line[0:2]))]

    print eqw_vals3

    writestr = 'element\tresidual\twavelength\n'
    for line in line_res1:
        writestr+=str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n'

    for line in line_res2:
        writestr+=str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n'

    for line in line_res3:
        writestr+=str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n'

    writestr += '\nelement\tequivalent width\twavelength\n'

    for line in eqw_vals1:
        writestr+=str(line[1])+'\t'+str(line[0])+'\t'+str(line[2])+'\n'

    for line in eqw_vals2:
        writestr+=str(line[1])+'\t'+str(line[0])+'\t'+str(line[2])+'\n'

    for line in eqw_vals3:
        writestr+=str(line[1])+'\t'+str(line[0])+'\t'+str(line[2])+'\n'


    print writestr

    outname = fitname.replace('.h5','')

    f = open(fitdir+outname+'_line_residuals.txt','w+')
    f.write(writestr)
    f.close()



