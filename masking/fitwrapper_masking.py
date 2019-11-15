import subprocess


import numpy as np
import os,scipy
from starkit.gridkit import load_grid

import sys

import multi_order_fitting_functions as mtf

def fitwrapper_sens_masking_three_order(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/'

    phoenix = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    bosz = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [60.0]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        #mtf.fit_star_three_orders_sens_masked(starname,'mh',phoenix,'phoenix',sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.,adderr=True,nirspec_upgrade=False)
        mtf.fit_star_three_orders_sens_masked(starname,'mh',bosz,'bosz',sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.,adderr=True,nirspec_upgrade=False)


def fitwrapper_sens_masking_three_order_phoe(starname):
    # cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/'

    # mod = load_grid('/u/rbentley/metallicity/grids/apogee_bosz_t3500_7000_w20000_24000_R25000.h5')
    mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    # for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [0.]:  # np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_three_orders_sens_masked(starname, 'mh', mod,'phoenix', sl_cut, specdir=spec_path, savedir=save_path,
                                              R_fixed=24000.,nirspec_upgrade=True)


def fitwrapper_sens_masking_four_order(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')
    #mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [0.0]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_four_orders_sens_masked(starname,'mh',mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)




def fitwrapper_sens_masking_four_order_non_mh_params(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/apogee_bosz_t3500_7000_w20000_24000_R25000.h5')

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

    #mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    mod = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for res_cut in np.arange(0.025,0.2,0.025): #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", res_cut
        mtf.fit_star_four_orders_residual_masked(starname,mod,res_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.)


def fitwrapper_res_masking_three_order(starname):
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/'

    # mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    mod = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

    # for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for res_cut in np.arange(0.025, 0.2, 0.025):  # np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", res_cut
        mtf.fit_star_three_orders_residual_masked(starname, mod,'bosz',res_cut, specdir=spec_path, savedir=save_path,
                                                 R_fixed=24000.)


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
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/order36/'

    mod = load_grid('/u/rbentley/metallicity/grids/apogee_bosz_t3600_7000_w21000_23000_R25000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in [0.]: #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_one_order_sens_masked(starname,'mh',mod,sl_cut,specdir=spec_path,savedir=save_path,teff_range=[3600,6000],logg_range=[0.,4.5],mh_range=[-1.,0.5],R_fixed=24000.,vrot_fixed=0.0)


def fitwrapper_fe_lines_masking_three_order(starname):
    # cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/'

    mod = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

    mtf.fit_star_three_orders_fe_lines(starname, mod, 'bosz', specdir=spec_path, savedir=save_path,
                                              R_fixed=24000., adderr=True,nirspec_upgrade=False)

def convolved_three_order(starname,grid):
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/'

    r = 5000.

    mtf.fit_star_three_orders_convolved(starname, grid,'bosz',r, specdir=spec_path, savedir=save_path,
                                                 R_fixed=r,adderr=True)


ngc6791_282 = 'NGC6791_J19205+3748282'

ngc6819_517 = 'NGC6819_J19411+4010517'

m5_0208 = 'M5 J15190+0208'

ngc6791_202 = 'NGC6791_J19213390+3750202'

ngc6819_482 = 'NGC6819_J19413439+4017482'

m71_021 = 'M71_J19534827+1848021'

tyc_3544 = 'TYC 3544'

m18113 = '2MJ18113-30441'

bosz = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

#fitwrapper_fe_lines_masking_three_order(ngc6791_282)
convolved_three_order(ngc6791_282,bosz)
convolved_three_order(ngc6819_517,bosz)
convolved_three_order(m5_0208,bosz)
convolved_three_order(ngc6791_202,bosz)
convolved_three_order(ngc6819_482,bosz)
convolved_three_order(m71_021,bosz)
convolved_three_order(tyc_3544,bosz)