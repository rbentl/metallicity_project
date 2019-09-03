import subprocess


import numpy as np
import os,scipy
from starkit.gridkit import load_grid

import sys

import multi_order_fitting_functions as mtf
'''
def fitwrapper_masking_singleorder():
    for i in np.concatenate((np.arange(0.,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
        print i
        proc = subprocess.Popen(['python','/u/rbentley/metallicity/code/masking/model_sensitivity_masked_fitting.py',str(i)])
        #print proc
        while True:
            if proc.poll() is not None:
                break
        #residual_mm = proc.returncode
        #print residual_mm


proc = subprocess.Popen(['python','model_s_masked_fitting.py',str(0.05)])
#print proc
while True:
    if proc.poll() is not None:
        break
'''

def fitwrapper_sens_masking_four_order(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [6.0]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_four_orders_sens_masked(starname,mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)




def fitwrapper_sens_masking_four_order_non_mh_params(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [6.0]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut, ' teff '+starname
        mtf.fit_star_four_orders_sens_masked(starname,'teff',mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)
        print "Starting fit with cut at ", sl_cut, ' logg '+starname
        mtf.fit_star_four_orders_sens_masked(starname,'logg',mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)
        print "Starting fit with cut at ", sl_cut, ' alpha '+starname
        mtf.fit_star_four_orders_sens_masked(starname,'alpha',mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)




def fitwrapper_res_masking_four_order(starname):
    
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for res_cut in np.concatenate((np.arange(0.02,0.06,0.01),np.arange(0.075,0.2,0.025))): #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", res_cut
        mtf.fit_star_four_orders_residual_masked(starname,mod,res_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)



def fitwrapper_masking_four_order_nirspec_upgrade(starname):
    
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [0]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_four_orders_sens_masked(starname,'mh',mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.,nirspec_upgrade=True,nsdrp_snr=True)


def fitwrapper_sens_masking_one_order(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/order35/'

    mod = load_grid('/u/rbentley/metallicity/grids/bosz_t3600_7000_w21000_23000_R25000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [0.]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_one_order_sens_masked(starname,'mh',mod,sl_cut,specdir=spec_path,savedir=save_path,teff_range=[3600,6000],logg_range=[0.,4.5],mh_range=[-1.,0.5],R_fixed=24000.,vrot_fixed=0.0)
        

ngc6791_282 = 'NGC6791_J19205+3748282'

ngc6819_517 = 'NGC6819_J19411+4010517'

m5_0208 = 'M5 J15190+0208'

ngc6791_202 = 'NGC6791_J19213390+3750202'

ngc6819_482 = 'NGC6819_J19413439+4017482'

m71_021 = 'M71_J19534827+1848021'

tyc_3544 = 'TYC 3544'

m18113 = '2MJ18113-30441'


fitwrapper_sens_masking_one_order(ngc6791_282)
#fitwrapper_sens_masking_four_order_non_mh_params(ngc6819_517)
#fitwrapper_sens_masking_four_order_non_mh_params(m5_0208)
#fitwrapper_sens_masking_four_order_non_mh_params(ngc6791_202)
#fitwrapper_sens_masking_four_order_non_mh_params(ngc6819_482)
#fitwrapper_sens_masking_four_order_non_mh_params(m71_021)
#fitwrapper_sens_masking_four_order_non_mh_params(tyc_3544)
