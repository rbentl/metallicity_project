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
import operator
import sys

import multi_order_fitting_functions as mtf

from multi_order_fitting_functions import Splitter3, Combiner3, apogee_vals, Splitter4, Combiner4


def getKey(item):
    return item[0]

def load_full_grid():
    g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
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

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/masked*.h5')

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

        for a in apogee_vals.keys():
            setattr(model,a,apogee_vals[a])

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


def plot_sl_response(sl_val,starname, plot_ref_vals=False):
    f= plt.figure(figsize=(12,11))
    rvax  = f.add_subplot(6,1,1)
    loggax  = f.add_subplot(6,1,2,sharex=rvax)
    mhax  = f.add_subplot(6,1,3)
    alphaax = f.add_subplot(6,1,4)
    teffax = f.add_subplot(6,1,5)
    lenax = f.add_subplot(6,1,6)

    
    sl_val = sorted(sl_val)
    #ax = plt.subplot()
    print [i[1] for i in sl_val]
    print [i[0] for i in sl_val]
    print len(sl_val)
    rvax.errorbar([i[0] for i in sl_val],[i[2] for i in sl_val],[i[10][5] for i in sl_val], color='red', fmt='o',linestyle='--',capsize=5,label='Order 34')
    rvax.errorbar([i[0] for i in sl_val],[i[3] for i in sl_val],[i[10][6] for i in sl_val], color='green', fmt='o',linestyle='--',capsize=5,label='Order 35')
    rvax.errorbar([i[0] for i in sl_val],[i[4] for i in sl_val],[i[10][7] for i in sl_val], color='blue', fmt='o',linestyle='--',capsize=5,label='Order 36')
    rvax.errorbar([i[0] for i in sl_val],[i[5] for i in sl_val],[i[10][8] for i in sl_val], color=0.4, fmt='o',linestyle='--',capsize=5,label='Order 37')

    rvax.set_xscale('log')
    rvax.set_ylabel('Radial Velocity (km/s)')
    rvax.set_title(starname+' masked fit responses')

    loggax.errorbar([i[0] for i in sl_val],[i[6] for i in sl_val],[i[10][1] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    loggax.set_xscale('log')
    loggax.set_ylabel('Log g')

    mhax.errorbar([i[0] for i in sl_val],[i[7] for i in sl_val],[i[10][2] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    mhax.set_xscale('log')
    mhax.set_ylabel('[M/H]')

    alphaax.errorbar([i[0] for i in sl_val],[i[8] for i in sl_val],[i[10][3] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    alphaax.set_xscale('log')
    alphaax.set_ylabel('$alpha$')

    teffax.errorbar([i[0] for i in sl_val],[i[9] for i in sl_val],[i[10][0] for i in sl_val], color='black',fmt='o',linestyle='--',capsize=5)
    teffax.set_xscale('log')
    teffax.set_ylabel('$T_{eff}$ (K)')

    lenax.plot([i[0] for i in sl_val],[i[1] for i in sl_val], color='black')
    lenax.axhline(y=970*4, color='k',linestyle='--', label='Total number of data points')
    lenax.set_xscale('log')
    lenax.set_xlabel('$S_{\lambda}$ cutoff')
    lenax.set_ylabel('# points masked')
    lenax.legend(loc='center right', fontsize=13)

    if plot_ref_vals and starname == 'NGC6791_J19205+3748282':
        rvax.axhline(y=-48.5, color='y',linestyle='--',label='APOGEE value')
        mhax.axhline(y=0.44, color='y',linestyle='--')
        teffax.axhline(y=4028, color='y',linestyle='--')
        alphaax.axhline(y=0.02, color='y',linestyle='--')
        loggax.axhline(y=1.5, color='y',linestyle='--')

    elif plot_ref_vals and starname == 'NGC6819_J19411+4010517':
        rvax.axhline(y=1.61, color='y',linestyle='--',label='APOGEE value')
        mhax.axhline(y=0.07, color='y',linestyle='--')
        teffax.axhline(y=4119, color='y',linestyle='--')
        alphaax.axhline(y=-0.01, color='y',linestyle='--')
        loggax.axhline(y=1.56, color='y',linestyle='--')

    elif plot_ref_vals and starname == 'M5 J15190+0208':
        rvax.axhline(y=61, color='y',linestyle='--',label='APOGEE value')
        mhax.axhline(y=-1.36, color='y',linestyle='--')
        teffax.axhline(y=3989, color='y',linestyle='--')
        alphaax.axhline(y=0.103, color='y',linestyle='--')
        loggax.axhline(y=0.72, color='y',linestyle='--')

        
    rvax.legend(bbox_to_anchor=(0.95,0.75), fontsize=13)
    plt.show()


def sl_response_plot_four(starname,g,specdir='/group/data/nirspec/spectra/',snr=30.,nnorm=2):

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

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/masked*'+starname+'_order34-37.h5')

    cut_lis = []

    for filename in h5_files_us:
        print filename.split('_')
        cut_lis += [(float(filename.split('_')[6]),filename)]

    cut_lis = sorted(cut_lis,key = getKey)

    h5_files = [i[1] for i in cut_lis]
    
    sl_val = []

    combined_data_mask_model = {}
    

    for filename in h5_files:
        print filename
        print filename.split('_')[6]
        gc_result = MultiNestResult.from_hdf5(filename)
        print gc_result

        for a in apogee_vals.keys():
            setattr(model,a,apogee_vals[a])

        w1,f1,w2,f2,w3,f3,w4,f4 = model()

        sl_mh1,sl_mh2,sl_mh3,sl_mh4 = mtf.s_lambda_four_order(model,'mh',model.mh_0.value,0.1)



        #combine all sl_mh,w,f, lists

        sl_mh = np.concatenate((sl_mh1,sl_mh2,sl_mh3,sl_mh4))

        w = np.concatenate((w1/(gc_result.median['vrad_3']/3e5+1.0),w2/(gc_result.median['vrad_4']/3e5+1.0),w3/(gc_result.median['vrad_5']/3e5+1.0),w4/(gc_result.median['vrad_6']/3e5+1.0)))

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

        print sigmas
        combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})
        sl_val += [(float(filename.split('_')[6]),len(mask_sl_f),gc_result.median['vrad_3'],gc_result.median['vrad_4'],gc_result.median['vrad_5'],gc_result.median['vrad_6'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],gc_result.median['teff_0'],sigmas)]
         
    print len(starfluxall)
    return sl_val, combined_data_mask_model


