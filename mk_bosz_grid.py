from starkit.gridkit.io import bosz
from starkit.gridkit.io.bosz.process import BOSZProcessGrid
from starkit.gridkit import load_grid
import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
from astropy import units as u
import os



def mk_bosz_grid(directory='/Volumes/data/bosz/',wave_range=[15000,17000],
                    teff_range=[2000,6000],logg_range=[0,5],mh_range=[-1.5,1.0],alpha_range=[-1,1],
                    R=40000,savefile=None):
    '''
    Read the raw bosz grid files and create a grid for Starkit

    Keywords
    --------
    directory - location of the bosz data and the database (bosz_grid_info.h5)
    wave_range - wavelength range to extract (in units of Angstrom) default: [22400,22750]
    teff_range - range in temperature (K) default: [2000,6000]
    logg_range - surface gravity range default: [0,5]
    mh_range - metallicity range default: [-1.5,1.0]
    alpha_range - range in alpha abundance default: [-1,1]
    R - resolution default: 20000
    savefile - file to save the grid, the default name has the temperature, wavelength range, 
               and resolution

    '''

    os.chdir(directory)
    meta = pd.read_hdf('bosz_grid_info.h5', 'meta')
    raw_index = pd.read_hdf('bosz_grid_info.h5', 'index')
    wavelength = pd.read_hdf('bosz_grid_info.h5', 'wavelength')[0].values * u.Unit(meta['wavelength_unit'])

    index_filter = (raw_index.teff.between(*teff_range) &
        raw_index.logg.between(*logg_range) &
        raw_index.mh.between(*mh_range) &
        raw_index.alpha.between(*alpha_range))

    new_index = raw_index.loc[index_filter]

    bgrid = BOSZProcessGrid(new_index, wavelength, meta,
                           wavelength_start=wave_range[0]*u.angstrom,
                           wavelength_stop=wave_range[1]*u.angstrom, R=R,
                          R_sampling=4)

    if savefile is None:
        savefile = '/u/rbentley/metallicity/grids/apogee_bosz_t%i_%i_w%i_%i_R%i.h5' % (int(teff_range[0]),int(teff_range[1]),
            int(wave_range[0]),int(wave_range[1]),int(R))

    bgrid.to_hdf(savefile, overwrite=True)



def mk_bosz_grid_k_nirspec_high():
    # make R = 40000 version of nirspec grid
    # make a bosz grid for K-band and NIRSPEC resolution
    mk_bosz_grid(wave_range=[20000,24000],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000)

def mk_bosz_grid_k_nirspec_high_34_36():
    # make R = 40000 version of nirspec grid
    # make a bosz grid for K-band and NIRSPEC resolution
    mk_bosz_grid(wave_range=[21000,22700],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000)


def mk_bosz_grid_h_apogee():
    # make R = 40000 version of nirspec grid
    # make a bosz grid for H-band and APOGEE
    mk_bosz_grid(wave_range=[15000,17000],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000)

def mk_bosz_grid_h_lower_R_apogee():
    # make R = 40000 version of nirspec grid
    # make a bosz grid for K-band and NIRSPEC resolution
    mk_bosz_grid(wave_range=[15000,17000],teff_range=[3500,6000],logg_range=[0,4.5],mh_range=[-1.0,0.5],alpha_range=[-0.25,0.5],R=25000)


def mk_bosz_grid_order35():
    # make R = 40000 version of nirspec grid
    # make a bosz grid for K-band and NIRSPEC resolution
    mk_bosz_grid(wave_range=[21000,23000],teff_range=[3600,7000],logg_range=[0.0,4.5],mh_range=[-1.0,0.5],alpha_range=[-0.23,0.5],R=25000)


def mk_bosz_grid_order35_trial():
    # make R = 40000 version of nirspec grid
    # make a bosz grid for K-band and NIRSPEC resolution
    mk_bosz_grid(wave_range=[21000,23000],teff_range=[3600,7000],logg_range=[0.5,4.5],mh_range=[-1.0,0.25],alpha_range=[-0.2,0.25],R=25000)
