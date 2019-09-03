import multi_order_fitting_functions as mff
import os
import glob
import scipy

fitpath = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36-37/'


os.chdir(fitpath)
h5files = glob.glob('*.h5')


cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))

passwd=raw_input('pwd: ')
for h5f in h5files:

    cut = float(h5f.split('_')[3])

    #orders = h5f.split('_')[5]

    for star in [x[0] for x in cal_star_info]:
        if star in h5f:
            star_ind = [x[0] for x in cal_star_info].index(star)
            starname = [x[0] for x in cal_star_info][star_ind]
    

    if cut == 0:
        method = 'unmasked'
    else:
        method = h5f.split('_')[1] 
    
    mff.update_mysql_db(starname, method, cut, 'PHOENIX', '34-37', h5f, fitpath, passwd)
