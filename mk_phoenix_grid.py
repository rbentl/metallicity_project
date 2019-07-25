from starkit.gridkit.io.phoenix import make_grid_info
from starkit.gridkit.io.phoenix import PhoenixProcessGrid
from starkit.gridkit import load_grid
import pandas as pd
from astropy import units as u, constants as const
from astropy.io import fits
import numpy as np
import uuid
import os

def mk_phoenix_grid(directory='/Volumes/nyx_backup/data/phoenix2016/',wave_range=[21500,22220],
                    teff_range=[2000,6000],logg_range=[0,5],mh_range=[-1.5,1.0],alpha_range=[-1,1],
                    R=40000,savefile=None):
    '''
    Read the raw phoenix grid files and create a grid for Starkit

    Keywords
    --------
    directory - location of the phoenix data and the database (phoenix_grid_info.h5)
    wave_range - wavelength range to extract (in units of Angstrom) default: [22400,22750]
    teff_range - range in temperature (K) default: [2000,6000]
    logg_range - surface gravity range default: [0,5]
    mh_range - metallicity range default: [-1.5,1.0]
    alpha_range - range in alpha abundance default: [-1,1]
    R - resolution default: 20000
    savefile - file to save the grid, the default name has the temperature, wavelength range, 
               and resolution

    '''
    # updated for new starkit version (2018)
    # for old phoenix grid, see /u/tdo/research/phoenix/
    os.chdir(directory)
    print os.listdir(directory)
    print os.getcwd()
    meta = pd.read_hdf('phoenix_grid_info.h5', 'meta')
    raw_index = pd.read_hdf('phoenix_grid_info.h5', 'index')
    wavelength = pd.read_hdf('phoenix_grid_info.h5', 'wavelength')[0].values * u.Unit(meta['wavelength_unit'])

    index_filter = (raw_index.teff.between(*teff_range) &
        raw_index.logg.between(*logg_range) &
        raw_index.mh.between(*mh_range) &
        raw_index.alpha.between(*alpha_range))

    new_index = raw_index.loc[index_filter]

    pgrid = PhoenixProcessGrid(new_index, wavelength, meta,
                           wavelength_start=wave_range[0]*u.angstrom,
                           wavelength_stop=wave_range[1]*u.angstrom, R=R,
                          R_sampling=4)

    #os.chdir('/u/rbentley/metallicity/spectra_fits')
    if savefile is None:
        savefile = '/u/rbentley/metallicity/spectra_fits/phoenix_t%i_%i_w%i_%i_R%i.h5' % (int(teff_range[0]),int(teff_range[1]),
            int(wave_range[0]),int(wave_range[1]),int(R))
    #os.chdir('/u/rbentley/metallicity/spectra_fits')
    print os.getcwd()
    pgrid.to_hdf(savefile, overwrite=True)

def mk_phoenix_grid_k_nirspec_high():
    # make R = 40000 version of nirspec grid
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[20000,24000],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000)


def mk_phoenix_grid_k_nirspec():
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[20000,24000],teff_range=[2000,6000],logg_range=[0,5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=25000
                    )
def mk_phoenix_grid_k_med_res():
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[20000,24000],teff_range=[2000,6000],logg_range=[0,5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=8000)


def mk_phoenix_grid_test():
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[20000,24000],teff_range=[2000,3000],logg_range=[0,1],mh_range=[0.5,1.0],alpha_range=[0,0.1],R=500)

def mk_phoenix_grid_k_nirspec_order34():
    # make R = 40000 version of nirspec grid
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[22350,22900],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000)

def mk_phoenix_grid_k_nirspec_order35():
    # make R = 40000 version of nirspec grid
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[21500,22200],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000)

def mk_phoenix_grid_k_nirspec_order36():
    # make R = 40000 version of nirspec grid
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[21000,21600],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000) #23500-24300

def mk_phoenix_grid_apogee():
    # make R = 40000 version of nirspec grid
    # make a phoenix grid for K-band and NIRSPEC resolution
    mk_phoenix_grid(wave_range=[15000,17000],teff_range=[2500,6000],logg_range=[0,4.5],mh_range=[-2.0,1.0],alpha_range=[-1,1],R=40000) #23500-24300
    
    
