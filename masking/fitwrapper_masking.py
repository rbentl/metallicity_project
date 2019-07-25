import subprocess


import numpy as np
import os,scipy
from starkit.gridkit import load_grid

import sys

import multi_order_fitting_functions as mtf

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

'''
proc = subprocess.Popen(['python','model_s_masked_fitting.py',str(0.05)])
#print proc
while True:
    if proc.poll() is not None:
        break
'''

def fitwrapper_masking_four_order(starname):
    

    #cal_star = 'NGC6819_J19413439+4017482'
    spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'

    mod = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')

    #for sl_cut in np.concatenate((np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):
    for sl_cut in np.concatenate((np.arange(0.4,2,0.2),np.arange(2,20,2),np.arange(20,400,20))): #np.concatenate((np.arange(60.,120.,15),np.arange(120.,390,45),np.arange(0.1,0.5,0.1),np.linspace(0.5,10.,20),np.arange(15.,60.,5))):

        print "Starting fit with cut at ", sl_cut
        mtf.fit_star_four_orders_masked(starname,mod,sl_cut,specdir=spec_path,savedir=save_path,R_fixed=24000.) 


cal_star_high_mh = 'NGC6791_J19205+3748282'

cal_star_mid_mh = 'NGC6819_J19411+4010517'

cal_star_low_mh = 'M5 J15190+0208'

#fitwrapper_masking_four_order(cal_star_high_mh)
fitwrapper_masking_four_order(cal_star_mid_mh)
fitwrapper_masking_four_order(cal_star_low_mh)
