import subprocess
import glob

import numpy as np
import os,scipy
from starkit.gridkit import load_grid
from astropy.io import ascii
from astroquery.vizier import Vizier
import sys

import multi_order_fitting_functions as mtf

def fitwrapper_spex(grid, gridname):

    specdir = '/u/tdo/research/metallicity/standards'

    star_specs = glob.glob(specdir+'/*.fits')

    for star_path in star_specs:

        starname = star_path.split('/')[-1]
        starname = starname.replace('.fits', '')

        print ('Beginning to fit field '+starname+' with the '+gridname+' grid.')

        spec_path = specdir+'/'

        save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/spex/'

        mtf.fit_spex_star(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True,waverange=[21000.,22910.])

def fitwrapper_nifs(dirname, grid, gridname):

    specdir = '/u/tdo/research/metallicity/spectra/'+dirname

    star_specs = glob.glob(specdir+'/*.fits')



    for star in star_specs:
        print star

        starname = star.split('/')[-1]
        starname = starname.replace('.fits', '')

        dash_star = starname.replace('_', '-')

        if (dash_star not in good_stars) & (dash_star not in ['NE-1-001', 'NE_1-002', 'NE-1-003']):
            print 'star ' + starname + 'is being ignored'
            continue

        print ('Beginning to fit field '+starname+' with the '+gridname+' grid.')

        spec_path = specdir+'/'

        save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/nifs/'

        mtf.fit_nifs_star_unmasked(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True,waverange=[21000.,22910.], logg_fixed=1.5)

def fitwrapper_three_order_koa(starname, grid, gridname):

    cal_mh = None

    if starname in ['E5_1_001','N2_1_002']:
        sn=15. #15
    elif starname in ['N2_1_001','N2_1_003']:
        sn=10. #10
    else:
        sn=30.

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[:-1]]
    if starname in cal_star_names:
        star_ind = cal_star_names.index(starname)
        cal_star_info = cal_star_info_all[star_ind]
        cal_mh = cal_star_info[1]

    print ('Beginning to fit star '+starname+' with the '+gridname+' grid. SNR = '+str(sn))

    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/koa_specs/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'

    mtf.fit_star_three_orders_unmasked(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True, logg_fixed=1.2)
    #mtf.fit_fire_star_three_orders_unmasked(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True, waverange=[15000.,17000.])
    #mtf.fit_star_three_orders_unmasked_no_rot_const_dl(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True,dl_fixed=0.961)


def fitwrapper_three_order(starname, grid, gridname):

    cal_mh = None

    if starname in ['E5_1_001','N2_1_002']:
        sn=15. #15
    elif starname in ['N2_1_001','N2_1_003']:
        sn=10. #10
    else:
        sn=30.

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[:-1]]
    if starname in cal_star_names:
        star_ind = cal_star_names.index(starname)
        cal_star_info = cal_star_info_all[star_ind]
        cal_mh = cal_star_info[1]

    print ('Beginning to fit star '+starname+' with the '+gridname+' grid. SNR = '+str(sn))

    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'

    mtf.fit_star_three_orders_unmasked(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True,logg_fixed=1.2,R_fixed=24000.)
    #mtf.fit_fire_star_three_orders_unmasked(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True, waverange=[15000.,17000.])
    #mtf.fit_star_three_orders_unmasked_no_rot_const_dl(starname,grid,gridname,specdir=spec_path,savedir=save_path,adderr=True,dl_fixed=0.961)



def fitplotter_three_order(starname, grid, gridname):

    print ('Beginning to plot star '+starname+' with the '+gridname+' grid.')

    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'

    fitp = save_path + starname + '_order34-36_'+gridname+'_adderr.h5'

    mtf.plot_fit_result_three_order(starname,fitp,grid, koa_spec=False)


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

    r = 4000.

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

phoenix = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
bosz = load_grid('/u/ghezgroup/data/metallicity/nirspec/grids/bosz_t3500_7000_w20000_24000_R50000.h5')

gc_stars = ['NE_1_001', 'NE_1_002', 'NE_1_003', 'E7_1_001', 'E7_2_001', 'E7_1_002', 'E7_1_003', 'N2_1_001', 'E5_1_001', 'N2_1_002', 'N2_1_003', 'S1-23']
cal_stars = ['NGC6791_J19205+3748282', 'NGC6819_J19411+4010517', 'M5 J15190+0208', 'NGC6791_J19213390+3750202', 'NGC6819_J19413439+4017482', 'M71_J19534827+1848021', 'TYC 3544']

fire_stars = ['HD176704', 'HD191584', 'HD221148', 'N6583_46']

nifs_fields = ['N1_1','E7_2','N1_2','N2_1','N2_2','NE_1', 'E5_1','E5_2','E6_1','E6_2','E7_1']

#Vizier.ROW_LIMIT = 100000
#catalog_list = Vizier.find_catalogs('Do, 2015')
#catalog = Vizier.get_catalogs(catalog_list.keys())['J/ApJ/808/106/stars']

#locs = np.where((catalog['SNR'] >= 20.))[0]

#good_stars = catalog['Name'][locs]

#gc_stars = ['N2_1_002', 'N2_1_003']
#fitwrapper_fe_lines_masking_three_order(ngc6791_282)

new_cals = ['M67_08512280+1148016', 'M67_08514388+1156425']

koa_stars = ['M71_J19534827+1848021','M71_J19534525+1846553', 'M71_19535325+1846471', 'M71_J19533757+1847286']

for star in gc_stars:
    fitwrapper_three_order(star, bosz, 'bosz')

spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'

#mtf.fit_star_three_orders_unmasked('M71_J19534827+1848021',bosz,'bosz',specdir=spec_path,savedir=save_path,adderr=True, R_fixed=24000.)

#fitwrapper_three_order(m18113, bosz, 'bosz')
#starname = 'NE_1_002'
#spec_path = '/u/tdo/research/metallicity/spectra/NE_1/'

#save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/nifs/'

#mtf.fit_nifs_star_unmasked(starname, bosz, 'bosz', specdir=spec_path, savedir=save_path, adderr=True,
#                           waverange=[21000., 22910.], logg_fixed=1.5)
