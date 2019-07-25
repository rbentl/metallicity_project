import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import units as u
from starkit.fitkit import likelihoods
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from specutils import read_fits_file,plotlines
import shutil, logging, datetime
import os,scipy
from specutils import Spectrum1D,rvmeasure
import subprocess
import gc
from astropy.io import fits
from astropy.time import Time
import glob

try:
    import MySQLdb as mdb
except:
    import pymysql as mdb


def update_starkit_db(name,date,ddate,mjd,h5file,snr=None,original_location=None,spectrum_file=None,
                      vlsr=None,passwd=None,vsys=0.0,source=None,vhelio=None,order=None):
    # update the starkit database with the info from the hdf5 file and the star
    print h5file
    if os.path.exists(h5file):
        results = MultiNestResult.from_hdf5(h5file)

        m = results.maximum
        med = results.median
        sig = results.calculate_sigmas(1)
        if 'add_err_6' in m.keys():
            p = ['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2','R_3','add_err_6']
        else:
            p = ['teff_0','logg_0','mh_0','alpha_0','vrot_1','vrad_2','R_3']
            
        temp = []
        for k in p:
            temp.append([m[k],med[k],(sig[k][1]-sig[k][0])/2.0,sig[k][1],sig[k][0]])

        values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
                 [original_location,spectrum_file,h5file,str(datetime.datetime.today())]

#        if 'add_err_6' in m.keys():
#            values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
#                     temp[7] + [original_location,spectrum_file,h5file,str(datetime.datetime.today())]
#        else:
#            values = [name,date,ddate,mjd]+ temp[0] + temp[1]+ temp[5] + temp[2] + temp[3] + temp[4] + temp[6] + \
#                     [None,None,None,None] + [original_location,spectrum_file,h5file,str(datetime.datetime.today())]


        if vlsr is None:
            values = values + [None,None]
        else:
            values = values + [temp[5][1]+vlsr,temp[5][0]+vlsr]

        if vhelio is None:
            values = values + [None,None]
        else:
            values = values + [temp[5][1]+vhelio,temp[5][0]+vhelio]

        inf = open('fitting_params_'+name+'.dat','r')
        lines = inf.readlines()
        for l in lines:
            lcont = l.split()
            if int(lcont[0]) == order:

                resmean = lcont[1]
                resmax = lcont[2]
                chi2 = lcont[3]
        inf.close()
        values = values + [vsys,snr,source]
        con = mdb.connect(host='galaxy1.astro.ucla.edu',user='dbwrite',passwd=passwd,db='gcg')
        cur = con.cursor()
        values = values + [None,None,None,None,None,None,None,None]
        values = values + [resmean,resmax,chi2,order,'PHOENIX']
        testval = values       
        print(values),len(values)
        sql_query = 'REPLACE INTO metallicity (name,date,ddate,mjd,'+\
                    'teff_peak,teff,teff_err,teff_err_upper,teff_err_lower,'+\
                    'logg_peak,logg,logg_err,logg_err_upper,logg_err_lower,'+\
                    'vz_peak,vz,vz_err,vz_err_upper,vz_err_lower,'+\
                    'mh_peak,mh,mh_err,mh_err_upper,mh_err_lower,'+\
                    'alpha_peak,alpha,alpha_err,alpha_err_upper,alpha_err_lower,'+\
                    'vrot_peak,vrot,vrot_err,vrot_err_upper,vrot_err_lower,'+\
                    'r_peak,r,r_err,r_err_upper,r_err_lower,'+\
                    'original_location,file,chains,edit,vlsr,vlsr_peak,vhelio,vhelio_peak,vsys_err,snr,source,simbad,simbad_err,add_err,add_err_peak,early_type,simbad_rot,simbad_rot_err,vsys_offset,residual_mean,residual_max,reduce_chi2,order_,grid) '+\
                    'VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'+\
                    '%s,%s,%s)'

        for x in range(len(values)):
            if type(values[x]) is np.float64:
               values[x] = float(values[x])
            print type(values[x])
               
        print('adding into db',type(values))
        #print(sql_query)
        #print(values)
        #print testval
        cur.execute(sql_query,values)
        #cur.execute('DELETE FROM starkit_nirspec;')
        con.commit()
        cur.close()
        con.close()


input_dir = '/u/ghezgroup/data/metallicity/nirspec/spectra_fits/bosz'

starnamelist = ['M5 J15190+0208','M71_J19534827+1848021','NGC6791_J19205+3748282','NGC6791_J19213390+3750202','NGC6819_J19411+4010517','NGC6819_J19413439+4017482','TYC 3544'] #BEGIN MAKING THIS
specdir = '/u/ghezgroup/data/metallicity/nirspec/spectra/'
orders = range(32,38)
radvlist = ['61.0','-20.16','-48.11','-46.76','1.61','2.93','0.0']
tefflist = [3989,4060.3,4082,3901.9,4118.5,4196.4,5556]
logglist = [0.72,1.08,1.5,1.26,1.56,1.75,1.0]
mhlist = [-1.26,-0.7,0.44,0.33,0.05,0.11,-0.02]
snrlist = [[6.852413776,6.780372426,41.60199371,71.46592798,43.63332212,63.96825864],
           [5.333482145,47.13041956,43.22173228,48.36096037,32.57623449],
           [3.694949348,21.25513934,23.41551156,23.17726831,20.39853108],
           [3.474242099,20.56052317,20.83144016,23.0123618,18.19729144],
           [4.59168933,25.93288951,31.49617065,37.23632494,30.31303672],
           [4.552741455,27.36461857,31.64449491,38.3520086,28.40172674],
           [26.25458582,24.91650192,23.85006037,28.60852157,51.00394526,53.66289803]]
q = 0
w = 0

#adjust to actual values
mjd = [57947.24791667,57527.62222222,57524.60416667,57527.58333333,57524.63055556,57527.61111111,57946.26111111] # 2018-09-16 UT
source = 'NIRSPEC'
vlsr = 0.0
vhelio = 0.0
vsys = 11.0
'''
for s in range(len(starnamelist)):
    
    w = 0
    snrange=np.array([2.14,2.15])*1e4
    #snrange = np.array([2.19,2.20])*1e4
    for o in range(len(orders)):
        order = str(orders[o])
        file2 = glob.glob(starnamelist[s]+'order'+order+'_test_results.h5')
        spectra = glob.glob(specdir+starnamelist[s]+'_order'+order+'*.fits')
        if file2 == []:
            continue
        else:
            proc = subprocess.Popen(['python','fitwrapper_multi_star.py',starnamelist[s],str(order),radvlist[s],str(tefflist[s]),str(logglist[s]),str(mhlist[s]),str(snrlist[s][w])])
            print "here test"
            while True:
                if proc.poll() is not None:
                    break
            #print starnamelist[s],str(order),radvlist[s],str(tefflist[s]),str(logglist[s]),str(mhlist[s]),str(snrlist[s][w])
            w += 1
            

    t1 = Time(mjd[s],format='mjd')
    year = str(t1.datetime.year)
    passwd = raw_input('pwd: ')
    for o in orders:
        order = str(o)
        file2 = glob.glob(starname+'order'+order+'_fixed_logg_mh.h5')
        spectra = glob.glob(specdir+starname+'_order'+order+'*.fits')
        
        if file2 == []:
            continue
        else:
            snr = snrlist[q]
            q += 1
    
            print(t1.datetime)
            update_starkit_db(starname,t1.datetime.date(),t1.jyear,t1.mjd,h5file=file2[0],original_location=spectra[0],spectrum_file=spectra[0],passwd=passwd,vlsr=vlsr,snr=snr,source=source,vsys=vsys,vhelio=vhelio,order=o)
'''
orders = range(34,37)
t1 = Time(mjd[-1],format='mjd')
year = str(t1.datetime.year)
passwd=raw_input('pwd: ')
tycsnr=[23.85006037,28.60852157,51.00394526]
for o in orders:
    order = str(o)
    print order
    file2 = glob.glob('masks/NGC6791_J19205+3748282order'+order+'_phoenixr40000_test2_o'+order+'.h5')
    spectra = glob.glob(specdir+'NGC6791_J19205+3748282_order'+order+'*.fits')
    print file2
    if file2 == []:
        continue
    else:
        snr = tycsnr[q]
        q += 1
    
        print(t1.datetime)
        update_starkit_db('NGC6791_J19205+3748282',t1.datetime.date(),t1.jyear,t1.mjd,h5file=file2[0],original_location=spectra[0],spectrum_file=spectra[0],passwd=passwd,vlsr=vlsr,snr=snr,source=source,vsys=vsys,vhelio=vhelio,order=o)
