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
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult

try:
    import MySQLdb as mdb
except:
    import pymysql as mdb

cal_stars = scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None)

cal_stars_init = sorted(cal_stars, key=lambda x: x[1])

gc_stars = scipy.genfromtxt('/u/rbentley/metallicity/gc_star_info.dat', delimiter='\t', skip_header=1,dtype=None)

gc_stars = sorted(gc_stars, key=lambda x: x[1])

order_info = scipy.genfromtxt('/u/rbentley/metallicity/reduced_orders.dat', delimiter='\t', skip_header=1,dtype=None)

cal_spec = []



for i in range(len(cal_stars_init)):
    for order in order_info:

        if order[0] == cal_stars_init[i][0] and order[1] == 'order35' and order[4] != 2:
            #print cal_stars_init[i]
            cal_spec += [[cal_stars_init[i][0], cal_stars_init[i][1], order[2]]]


for i in range(len(cal_stars_init)):
    fitting_res = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/'+cal_stars_init[i][0]+'order35_test_results.h5')
    print fitting_res
    print fitting_res.median['vrad_2']
    cal_spec[i] += [fitting_res.median['vrad_2']]
    
            
            
'''
passwd=raw_input('Sql pwd: ')
con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbread',passwd=passwd,db='gcg')
cur = con.cursor()

sql_query = 'SELECT name, date, teff_peak, teff, teff_err,' +\
            'logg_peak,logg,logg_err,logg_err_upper,logg_err_lower,'+\
            'vz_peak,vz,vz_err,vz_err_upper,vz_err_lower,'+\
            'mh_peak,mh,mh_err,mh_err_upper,mh_err_lower,'+\
            'alpha_peak,alpha,alpha_err,alpha_err_upper,alpha_err_lower,'+\
            'order_ FROM metallicity'

cur.execute(sql_query)

database_vals = cur.fetchall()
cur.close()
con.close()
'''
print cal_spec

for star in cal_spec:
    specdir='/u/rbentley/metallicity/spectra/'
    print specdir+star[0]
    files = glob.glob(specdir+star[0]+'_order35*.dat')
    print files
    snr = star[3]
    starspectrum = read_fits_file.read_nirspec_dat(files,desired_wavelength_units='micron')
    waverange = [np.amin(starspectrum.wavelength.value[:970]), np.amax(starspectrum.wavelength.value[:970])]

    starspectrum = read_fits_file.read_nirspec_dat(files,desired_wavelength_units='Angstrom',wave_range=waverange)
    starspectrum.uncertainty = (np.zeros(len(starspectrum.flux.value))+1.0/np.float(snr))*starspectrum.flux.unit

    star += [starspectrum]
    

calfig, calax  = plt.subplots(len(cal_spec),sharex=True)

print len(cal_spec)

for i in range(len(calax)):
    print cal_spec[i][2], str(cal_spec[i][0])
    calax[i].plot(cal_spec[i][4].wavelength.value/(cal_spec[i][2]/3e5 +1.),cal_spec[i][4].flux.value)
    calax[i].set_ylabel('[M/H]' + str(cal_spec[i][1]))

plt.show()
