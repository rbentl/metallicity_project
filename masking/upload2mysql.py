import multi_order_fitting_functions as mff
import os
import glob
import scipy

fitpath = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/PHOENIX_fits/res_masked/'


os.chdir(fitpath)
h5files = glob.glob('masked*.h5')

#phoenix = mff.load_full_grid_phoenix()
#bosz = mff.load_full_grid_bosz()

cal_star_info = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))

id = 1

passwd='w3p6QXyYx' #raw_input('pwd: ')

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
    
    #mff.update_mysql_db(starname, method, cut, bosz,'BOSZ', '34-36', h5f, fitpath, passwd)
    mff.add_column_mysql_db(h5f, fitpath, passwd, id)
    id += 1
