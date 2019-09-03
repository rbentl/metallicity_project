import sl_response_plot_multiorder as slp
from specutils import read_fits_file,plotlines
import pylab as plt
import os,scipy


cal_star_init = list(scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1,dtype=None))
cal_star_info = sorted(cal_star_init, key=lambda x: x[1])
cal_star_names = [x[0] for x in cal_star_info]

print cal_star_names, cal_star_info[1]

spec_path = '/u/rbentley/metallicity/spectra/'

mod = slp.load_full_grid()

sl_val_list = []
res_val_list = []
starname_list = []

for starname in cal_star_names:
    sl_val, sl_data = slp.sl_response_plot_four(starname, mod, specdir = spec_path)
    res_val, res_data = slp.residual_masked_param_info(starname, mod, specdir = spec_path)
    #slp.plot_sl_res_response(sl_val,res_val,starname, savefig=True)

    sl_val_list += [sl_val]
    res_val_list += [res_val]
    starname_list += [starname]


slp.plot_sl_res_response_allstar(sl_val_list,res_val_list,starname_list, savefig=True)
