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
from matplotlib.pyplot import cm
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

    h5_files_us = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/masked_sl*'+starname+'_order34-37.h5')

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

        print sigmas
        #combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})

        combined_data_mask_model.update({filename.split('_')[6] : [(starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value),\
                                                                   (starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value),\
                                                                   (starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value),\
                                                                   (starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0),starspectrum37.flux.value),\
                                                                   (w1,f1),(w2,f2),(w3,f3),(w4,f4),(mask_sl_w,mask_sl_f)]})
        
        sl_val += [(float(filename.split('_')[6]),len(mask_sl_f),gc_result.median['vrad_3'],gc_result.median['vrad_4'],gc_result.median['vrad_5'],gc_result.median['vrad_6'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],gc_result.median['teff_0'],sigmas)]
         
    print len(starfluxall)
    return sl_val, combined_data_mask_model




def residual_masked_param_info(starname,g,specdir='/group/data/nirspec/spectra/',snr=30.,nnorm=2):

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
        print filename
        print filename.split('_')[6]
        gc_result = MultiNestResult.from_hdf5(filename)
        print gc_result

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

        print sigmas
        #combined_data_mask_model.update({filename.split('_')[6] : [(starwaveall,starfluxall),(w,f),(mask_sl_w,mask_sl_f)]})

        combined_data_mask_model.update({filename.split('_')[6] : [(starspectrum34.wavelength.value/(gc_result.median['vrad_3']/3e5+1.0),starspectrum34.flux.value),\
                                                                   (starspectrum35.wavelength.value/(gc_result.median['vrad_4']/3e5+1.0),starspectrum35.flux.value),\
                                                                   (starspectrum36.wavelength.value/(gc_result.median['vrad_5']/3e5+1.0),starspectrum36.flux.value),\
                                                                   (starspectrum37.wavelength.value/(gc_result.median['vrad_6']/3e5+1.0),starspectrum37.flux.value),\
                                                                   (w1,f1),(w2,f2),(w3,f3),(w4,f4),(residual_masked_wavelength,residual_masked_flux)]})
        
        res_val += [(float(filename.split('_')[6]),len(residual_masked_flux),gc_result.median['vrad_3'],gc_result.median['vrad_4'],gc_result.median['vrad_5'],gc_result.median['vrad_6'],gc_result.median['logg_0'],gc_result.median['mh_0'],gc_result.median['alpha_0'],gc_result.median['teff_0'],sigmas)]
         
    print len(starfluxall)
    return res_val, combined_data_mask_model


def plot_sl_res_response(sl_val,res_val,starname, savefig=False):
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



def plot_sl_res_response_allstar(sl_val_list,res_val_list,starname_list, savefig=False):
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

        print sl_val[0][0]
    
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
        sl_loggax.set_ylabel('Log g difference',fontsize=10)

        sl_mhax.errorbar([i[0] for i in sl_val],[i[7]-float(cal_star_info[1]) for i in sl_val],[i[10][2] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_mhax.set_ylabel('[M/H] difference',fontsize=10)

        sl_alphaax.errorbar([i[0] for i in sl_val],[i[8]-float(cal_star_info[4]) for i in sl_val],[i[10][3] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_alphaax.set_ylabel('$alpha$ difference',fontsize=10)

        sl_teffax.errorbar([i[0] for i in sl_val],[i[9]-float(cal_star_info[2]) for i in sl_val],[i[10][0] for i in sl_val], c=c,fmt='o',linestyle='--',capsize=5)
        sl_teffax.set_ylabel('$T_{eff}$ (K) difference',fontsize=10)

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
        res_loggax.set_ylabel('Log g difference',fontsize=10)
    

        res_mhax.errorbar([i[0] for i in res_val],[i[7]-float(cal_star_info[1]) for i in res_val],[i[10][2] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5)
        res_mhax.set_ylabel('[M/H] difference',fontsize=10)

        res_alphaax.errorbar([i[0] for i in res_val],[i[8]-float(cal_star_info[4]) for i in res_val],[i[10][3] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5)
        res_alphaax.set_ylabel('$alpha$ difference',fontsize=10)

        res_teffax.errorbar([i[0] for i in res_val],[i[9]-float(cal_star_info[2]) for i in res_val],[i[10][0] for i in res_val], c=c,fmt='o',linestyle='--',capsize=5)
        res_teffax.set_ylabel('$T_{eff}$ difference(K)',fontsize=10)

        #res_lenax.plot([i[0] for i in res_val],[i[1] for i in res_val], color='black')

        #res_lenax.set_xlabel('Residual cutoff',fontsize=12)
        #res_lenax.set_ylabel('# points masked',fontsize=12)
        res_loggax.legend(loc='center right', fontsize=9)
    
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
