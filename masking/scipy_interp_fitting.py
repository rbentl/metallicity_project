from starkit.gridkit import load_grid
from scipy.interpolate import LinearNDInterpolator
import scipy
import numpy as np
import multi_order_fitting_functions as mtf
from matplotlib import pyplot as plt

grid = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

#grid_2 = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

points, values, wavelengths = grid.index, grid.fluxes.value, grid.wavelength.value

for i in range(len(points)):
    fit = np.polyfit(wavelengths, values[i], 1)
    values[i] = values[i] / (fit[0] * wavelengths + fit[1])

test_point1 = np.where(np.all(points == np.array([3500., 1.5, -1.0, -0.25]), axis=1))

test_point2 = np.where(np.all(points == np.array([3500., 1.5, -1.0, 0.25]), axis=1))

test_point3 = np.where(np.all(points == np.array([3500., 1.5, -1.0, 0.0]), axis=1))

test_point4 = np.where(np.all(points == np.array([3750., 1.5, -1.0, 0.0]), axis=1))

test_point_cal = np.where(np.all(points == np.array([3750., 1.5, 0.5, 0.0]), axis=1))

test_point_low_mh_gc = np.where(np.all(points == np.array([4250., 1.5, -1.0, -0.25]), axis=1))

test_point_high_mh_gc = np.where(np.all(points == np.array([3500., 1.5, 0.75, 0.25]), axis=1))

test_pts = [test_point1, test_point2, test_point3, test_point4, test_point_cal, test_point_low_mh_gc, test_point_high_mh_gc]

nirspec_range_pts = [test_point_cal, test_point_low_mh_gc, test_point_high_mh_gc]

#points[:, 0] = np.log10(points[:, 0])

worst_fit_pt = np.where(np.all(points == np.array([3500., 1.0, -2.0, 0.0]), axis=1))
worst_fit_pt_bound1 = np.where(np.all(points == np.array([3750., 1.0, -2.0, 0.0]), axis=1))
worst_fit_pt_bound2 = np.where(np.all(points == np.array([3500., 1.5, -2.0, 0.0]), axis=1))
worst_fit_pt_bound3 = np.where(np.all(points == np.array([3500., 1.0, -1.5, 0.0]), axis=1))
worst_fit_pt_bound4 = np.where(np.all(points == np.array([3500., 1.0, -2.0, 0.25]), axis=1))
worst_fit_pt_bound5 = np.where(np.all(points == np.array([3500., 1.0, -2.0, -0.25]), axis=1))
worst_fit_pt_bound6 = np.where(np.all(points == np.array([3500., 0.5, -2.0, 0.0]), axis=1))

worst_pts = [worst_fit_pt, worst_fit_pt_bound1, worst_fit_pt_bound2, worst_fit_pt_bound3, worst_fit_pt_bound5, worst_fit_pt_bound6]

nirspec_points = np.where((wavelengths >= 21100) & (wavelengths <= 22300))[0]

print worst_pts

residual_specs = []
model_specs = []
waves = []
names = []
grid_specs = []
points, values, wavelengths = grid.index.copy(), grid.fluxes.value.copy(), grid.wavelength.value.copy()


for i in worst_pts:
    #grid = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

    #points, values, wavelengths = grid.index, grid.fluxes.value, grid.wavelength.value

    #for i in range(len(points)):
    #    fit = np.polyfit(wavelengths, values[i], 1)
    #    values[i] = values[i] / (fit[0] * wavelengths + fit[1])

    removed_points = points[i][0]
    removed_flux = values[i][0]

    grid.fluxes.value = np.delete(values, i)
    grid.index = np.delete(points, i)

    print removed_points
    print removed_flux

    #interp_test = LinearNDInterpolator(np.delete(points, i, axis=0), np.delete(values, i, axis=0))


    #interp_spec = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    #print interp_spec, values[i]

    #save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    save_path = '/u/rbentley/plots for meetings/012420/starkit_interpolator/'
    fname = 'T:'+str(removed_points[0])+' log g:'+str(removed_points[1])+' [M/H]:'+str(removed_points[2])+r' \alpha:'+str(removed_points[3])
    save_name = save_path+fname+'_starkit_eval.png'

    model_spec, res_spec, wave = mtf.get_starkit_model(removed_flux, wavelengths, grid, ref_values=removed_points)#, savedir=save_path)

    residual_specs += [res_spec]
    model_specs += [model_spec]
    waves += [wave]
    names += [fname]
    grid_specs += [removed_flux]

    #points = grid.fluxes.value

mtf.plot_multiple_models(grid_specs, model_specs, waves, residual_specs, names)

'''
for i in test_pts:
    removed_points = points[i][0]
    removed_flux = values[i][0]
    print removed_points
    print removed_flux

    interp_test = LinearNDInterpolator(np.delete(points, i, axis=0), np.delete(values, i, axis=0))


    interp_spec = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    #print interp_spec, values[i]

    #save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    save_path = '/u/rbentley/plots for meetings/012420/starkit_interpolator/'
    fname = 'bosz_t'+str(removed_points[0])+'_logg'+str(removed_points[1])+'_mh'+str(removed_points[2])+'_a'+str(removed_points[3])+'_spec'
    save_name = save_path+fname+'_starkit_eval.png'

    mtf.plot_txt_files(save_name, removed_flux, wavelengths, grid_2, ref_values=removed_points)#, savedir=save_path)


for i in nirspec_range_pts:
    removed_points = points[i][0]
    #print removed_points
    interp_test = LinearNDInterpolator(np.delete(points, i, axis=0), np.delete(values, i, axis=0))

    interp_spec = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    fit_wavelength = np.array([wavelengths[x] for x in nirspec_points])
    fit_flux = np.array([interp_spec[x] for x in nirspec_points])

    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    fname = 'bosz_t'+str(removed_points[0])+'_logg'+str(removed_points[1])+'_mh'+str(removed_points[2])+'_a'+str(removed_points[3])+'_nirspec_range_spec'
    saved_name = save_path+fname+'_starkit_fit.h5'
    mtf.plot_txt_files(saved_name, interp_spec, wavelengths, grid_2, ref_values=removed_points)#, savedir=save_path)

'''
'''
#plotted_f = max_res_f #/np.median(max_res_f[plot_points])
#plotted_interp_f = teff_interpolated_f #/np.median(teff_interpolated_f[plot_points])

median = np.median(plotted_f)

teff_interpolated_res = plotted_f - plotted_interp_f

plt.plot(wavelengths,plotted_f,color='#000000', label='BOSZ Spectrum', linewidth=2.0)
plt.plot(wavelengths,plotted_interp_f, color='#33AFFF', label='Interpolated fit/residuals', linewidth=2.0)
plt.plot(wavelengths,teff_interpolated_res, color='#33AFFF', linewidth=2.0)



plt.axhline(y=0.05*median, color='k', linestyle='--', label='$\pm$ 5%')
plt.axhline(y=-0.05*median, color='k', linestyle='--')

plt.legend()
plt.title('BOSZ grid and $T_{eff}$ interpolated spectrum comparison at\n highest max fractional residual spectrum (T=3500, logg=1.5, [M/H]=0.75., [alpha/Fe]=0.25)')

#plt.xlim(21900,22100)
plt.ylabel('Normalized Flux')
plt.xlabel('Wavelength (Angstroms)')
plt.show()
'''



#max_r_f = values[i][0]

#f = open('/u/rbentley/metallicity/scipy_max_fres_spec.dat', 'w+')

#f.write('flux\twavelength\n')
#for i in range(len(max_r_f)):
#    f.write(str(max_r_f[i]) + '\t' + str(wavelengths[i]) + '\n')
#f.close()

#save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
#fname = 'scipy_max_fres_spec' #'/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/bosz_t3500_logg1.5_mh0.5_a0._spec_starkit_fit.h5'
#mtf.fit_txt_files(fname, max_r_f, wavelengths, grid, savedir=save_path)





'''

samp_spec_idx = np.where(np.all(points == np.array([3750., 1.5, 0.5, 0.]), axis=1))

for i in samp_spec_idx:
    #interp_test = LinearNDInterpolator(np.delete(points,i,axis=0), np.delete(values,i,axis=0))
    #max_r_f = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    max_r_f = values[i][0]

    print max_r_f

    print wavelengths

    #f = open('/u/rbentley/metallicity/bosz_t3750_logg1.5_mh0.5_a0._spec.dat', 'w')

    #f.write('flux\twavelength\n')

    #for i in range(len(max_r_f)):
    #    f.write(str(max_r_f[i]) + '\t' + str(wavelengths[i]) + '\n')
    #f.close()

    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    fname = 'bosz_t3750_logg1.5_mh0.5_a0._spec'
    #mtf.fit_txt_files(fname, max_r_f, wavelengths, grid, savedir=save_path)
    mtf.plot_txt_files(save_path+'bosz_t3750_logg1.5_mh0.5_a0._spec_starkit_fit.h5', max_r_f, wavelengths, grid)
'''
#samp_spec2_idx = np.where(np.all(points == np.array(3500., 1.5, -1., 0.]), axis=1))


#low_mh_flux = values[samp_spec2_idx][0]


#save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
#fname = 'bosz_t4250_logg1.5_mh-1._a-0._spec_starkit_fit'
#mtf.fit_txt_files(fname, max_r_f, wavelengths, grid, savedir=save_path)
#mtf.plot_txt_files(save_path+'bosz_t4250_logg1.5_mh-1._a-0._spec_starkit_fit_starkit_fit.h5', low_mh_flux, wavelengths, grid)


