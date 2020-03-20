from starkit.gridkit import load_grid
from scipy.interpolate import LinearNDInterpolator
import scipy
import numpy as np
import multi_order_fitting_functions as mtf

grid = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

points, values, wavelengths = grid.index, grid.fluxes.value, grid.wavelength.value

f = open('/u/rbentley/metallicity/scipy_interp_residuals_logteff.txt', 'w')

f.write('teff\tlog g\t[M/H]\t[alpha/Fe]\t')
f.write('max res\tmean res\tmax frac res\tmean frac res\t')

points[:, 0] = np.log10(points[:, 0])

for i in range(len(points)):
    fit = np.polyfit(wavelengths, values[i], 1)
    print fit
    values[i] = values[i] / (fit[0] * wavelengths + fit[1])
    print values[i], (fit[0] * wavelengths + fit[1])

for i in range(len(points)):
    removed_points = points[i]

    interp_test = LinearNDInterpolator(np.delete(points, i, axis=0), np.delete(values, i, axis=0))
    print interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    max_res = np.absolute((interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) - values[i])).max()
    mean_res = np.mean(interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) - values[i])
    max_relative = np.absolute(((interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) - values[i]) / values[i])).max()
    mean_relative = np.mean(((interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) - values[i]) / values[i]))

    for param in removed_points:
        f.write(str(param) + '\t')

    f.write(str(max_res) + '\t' + str(mean_res) + '\t' + str(max_relative) + '\t' + str(mean_relative) + '\n')

f.close()
'''
f = open('/u/rbentley/metallicity/scipy_interp_residuals_log_teff_normalized.txt', 'w')

f.write('log teff\tlog g\t[M/H]\t[alpha/Fe]\t')
f.write('max res\tmean res\tmax frac res\tmean frac res\t')

points[:, 0] = np.log10(points[:, 0])

for i in range(len(points)):
    removed_points = points[i]

    interp_test = LinearNDInterpolator(np.delete(points, i, axis=0), np.delete(values, i, axis=0))

    max_res = np.absolute(
        (interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) - values[i])).max()
    mean_res = np.mean(
        interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) - values[i])
    max_relative = np.absolute(((interp_test(removed_points[0], removed_points[1], removed_points[2],
                                             removed_points[3]) - values[i]) / values[i])).max()
    mean_relative = np.mean(((interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3]) -
                              values[i]) / values[i]))

    print max_res, mean_res
    print max_relative, mean_relative

    for param in removed_points:
        f.write(str(param) + '\t')

    f.write(str(max_res) + '\t' + str(mean_res) + '\t' + str(max_relative) + '\t' + str(mean_relative) + '\n')

f.close()
'''



'''

max_frac_res_idx = np.where(np.all(points == np.array([3500., 1., -2., 0.]), axis=1))

for i in max_frac_res_idx:
    removed_points = points[i][0]
    print removed_points, i
    interp_test = LinearNDInterpolator(np.delete(points,i,axis=0), np.delete(values,i,axis=0))
    max_r_f = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    f = open('/u/rbentley/metallicity/scipy_max_fres_spec.dat', 'w+')

    f.write('flux\twavelength\n')

    for i in range(len(max_r_f)):
        f.write(str(max_r_f[i]) + '\t' + str(wavelengths[i]) + '\n')
    f.close()

    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    fname = 'scipy_max_fres_spec'
    mtf.fit_txt_files(fname, max_r_f, wavelengths, grid, savedir=save_path)







samp_spec_idx = np.where(np.all(points == np.array([3750., 1.5, 0.5, 0.]), axis=1))

for i in samp_spec_idx:
    removed_points = points[i][0]
    print removed_points, i
    interp_test = LinearNDInterpolator(np.delete(points,i,axis=0), np.delete(values,i,axis=0))
    max_r_f = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    f = open('/u/rbentley/metallicity/bosz_t3500_logg1.5_mh0.5_a0._spec.dat', 'w+')

    f.write('flux\twavelength\n')

    for i in range(len(max_r_f)):
        f.write(str(max_r_f[i]) + '\t' + str(wavelengths[i]) + '\n')
    f.close()

    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    fname = 'bosz_t3500_logg1.5_mh0.5_a0._spec'
    mtf.fit_txt_files(fname, max_r_f, wavelengths, grid, savedir=save_path)


samp_spec2_idx = np.where(np.all(points == np.array([5000., 2.5, -0.5, 0.]), axis=1))

for i in samp_spec2_idx:
    removed_points = points[i][0]
    print removed_points, i
    interp_test = LinearNDInterpolator(np.delete(points,i,axis=0), np.delete(values,i,axis=0))
    max_r_f = interp_test(removed_points[0], removed_points[1], removed_points[2], removed_points[3])

    f = open('/u/rbentley/metallicity/bosz_t5000_logg2.5_mh-0.5_a0._spec.dat', 'w+')

    f.write('flux\twavelength\n')

    for i in range(len(max_r_f)):
        f.write(str(max_r_f[i]) + '\t' + str(wavelengths[i]) + '\n')
    f.close()

    save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    fname = 'bosz_t5000_logg2.5_mh-0.5_a0._spec'
    mtf.fit_txt_files(fname, max_r_f, wavelengths, grid, savedir=save_path)

'''


'''
grid_residuals = list(scipy.genfromtxt('/u/rbentley/metallicity/scipy_interp_residuals.txt', delimiter='\t', skip_header=1, dtype=None))

max_res = [x[1] for x in grid_residuals]
mean_res = [abs(x[2]) for x in grid_residuals]

max_relres = [x[3] for x in grid_residuals]
mean_relres = [abs(x[4]) for x in grid_residuals]

min_maxres = np.where(max_res == np.nanmax(max_res))
min_meanres = np.where(mean_res == np.nanmax(mean_res))

min_max_relres = np.where(max_relres == np.nanmax(max_relres))
min_mean_relres = np.where(mean_relres == np.nanmax(mean_relres))

print min_maxres, min_meanres, min_max_relres, min_mean_relres

print 'Smallest maximum residual at ', str(grid_residuals[min_maxres[0]])
print 'Smallest mean residual at ', str(grid_residuals[min_meanres[0]])
print 'Smallest maximum fractional residual at ', str(grid_residuals[min_max_relres[0]])
print 'Smallest mean fractional residual at ', str(grid_residuals[min_mean_relres[0]])
'''



