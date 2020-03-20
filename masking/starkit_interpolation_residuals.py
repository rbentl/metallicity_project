from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood
from starkit.fitkit.likelihoods import SpectralL1Likelihood as L1Likelihood
from starkit.fitkit.likelihoods import SpectralChi2LikelihoodAddErr as Chi2LikelihoodAddErr
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest,MultiNestResult
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from starkit import assemble_model, operations
from starkit.fitkit import priors
from specutils import read_fits_file,plotlines,write_spectrum
from scipy.interpolate import LinearNDInterpolator
import scipy
import numpy as np
from matplotlib import pyplot as plt
from specutils import Spectrum1D,rvmeasure
import astropy.units as u
from astropy.modeling import models,fitting
from astropy.modeling import Model
import matplotlib.pyplot as plt
from specutils import read_fits_file,plotlines,write_spectrum
import multi_order_fitting_functions as mtf


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


grid = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w20000_24000_R25000.h5')

points, values, wavelengths = grid.index.copy(), grid.fluxes.value.copy(), grid.wavelength.value.copy()

idx = np.where(np.all(points == np.array([3500., 0.0, -1.25, 0.0]), axis=1))[0]

idx2 = np.where(np.all(points == np.array([3500., 0.0, -0.75, 0.0]), axis=1))[0]


k = open('/u/rbentley/starkit_interpol_residuals/bosz_eq_widths_wider_range.dat', 'w')
k.write('teff\tlogg\t[M/H]\t[alpha/Fe]\tSi22078 EW\tSi22078 cont\tSi21378 EW\tSi21378 cont\tS22574 EW\tSi22574 cont\tFe22469 EW\tFe22469 cont\tFe22479 EW\tFe22479 cont\tFe22500 EW\tFe22500 cont\n')

for i in range(len(points)):
    fit = np.polyfit(wavelengths, values[i], 1)
    values[i] = values[i] / (fit[0] * wavelengths + fit[1])

for i in range(len(points)):

    removed_points = points[i]
    removed_flux = values[i]

    print removed_points
    print removed_flux

    grid.fluxes = np.delete(values, i, axis=0)
    grid.index = np.delete(points, i, axis=0)

    print len(grid.index)


    starspectrum = Spectrum1D.from_array(dispersion=wavelengths, flux=removed_flux,
                                           dispersion_unit=u.angstrom,
                                           uncertainty=removed_flux*(1/100.))

    interp1 = Interpolate(starspectrum)
    norm1 = Normalize(starspectrum, 2)

    model = grid | interp1 | norm1

    setattr(model, 'teff_0', removed_points[0])
    setattr(model, 'logg_0', removed_points[1])
    setattr(model, 'mh_0', removed_points[2])
    setattr(model, 'alpha_0', removed_points[3])
    '''
    result = mtf.fit_array(starspectrum, model, R_fixed=25000.)

    print result.median

    for a in result.median.keys():
        setattr(model, a, result.median[a])

    '''

    w1, f1 = model()

    res1 = starspectrum.flux.value - f1

    data_idx = np.where((w1 >= 21150) & (w1 <= 22750))[0]

    data_w1 = w1[data_idx]


    #data = list(scipy.genfromtxt('/u/rbentley/starkit_interpol_residuals_log_teff/bosz_t' + str(removed_points[0]) + '_logg' + str(
    #    removed_points[1]) + '_mh' + str(removed_points[2]) + '_a' + str(removed_points[3]) + '_log_teff.dat', delimiter='\t', skip_header=1, dtype=None))

    #res1 = [x[3] for x in data]
    #w1 = [x[0] for x in data]




    #max_r_idx, max_r = np.argmax(res1), np.amax(res1)

    #data_max_r, data_max_r_w = np.amax(res1[data_idx]), data_w1[np.argmax(res1[data_idx])]

    #data_idx_34 = np.where((w1 >= 22400) & (w1 <= 22750))[0]
    #data_idx_35 = np.where((w1 >= 21750) & (w1 <= 22100))[0]
    #data_idx_36 = np.where((w1 >= 21150) & (w1 <= 21500))[0]

    data_idx_si1 = np.where((w1 >= 22065) & (w1 <= 22085))[0]
    data_idx_si2 = np.where((w1 >= 21355) & (w1 <= 21415))[0]
    data_idx_s1 = np.where((w1 >= 22490) & (w1 <= 22520))[0]

    data_idx_fe1 = np.where((w1 >= 22450) & (w1 <= 22490))[0]
    data_idx_fe2 = np.where((w1 >= 22460) & (w1 <= 22510))[0]
    data_idx_fe3 = np.where((w1 >= 22485) & (w1 <= 22515))[0]

    #cont34 = np.amax(starspectrum.flux.value[data_idx_34])
    #cont35 = np.amax(starspectrum.flux.value[data_idx_35])
    #cont36 = np.amax(starspectrum.flux.value[data_idx_36])

    cont35_si1 = np.amax(starspectrum.flux.value[data_idx_si1])
    cont36_si2 = np.amax(starspectrum.flux.value[data_idx_si2])
    cont34_s1 = np.amax(starspectrum.flux.value[data_idx_s1])

    cont34_fe1 = np.amax(starspectrum.flux.value[data_idx_fe1])
    cont34_fe2 = np.amax(starspectrum.flux.value[data_idx_fe2])
    cont34_fe3 = np.amax(starspectrum.flux.value[data_idx_fe3])

    si1_ew = rvmeasure.equivalent_width(starspectrum.wavelength.value, starspectrum.flux.value, specRange=[22076.5, 22080.5],
                                     continuum=cont35_si1)  # Si line

    si2_ew = rvmeasure.equivalent_width(starspectrum.wavelength.value, starspectrum.flux.value, specRange=[21376.5, 21380.5],
                                     continuum=cont36_si2)  # Si line
    s1_ew = rvmeasure.equivalent_width(starspectrum.wavelength.value, starspectrum.flux.value, specRange=[22511.5, 22516],
                                     continuum=cont34_s1)  # S line

    fe1_ew = rvmeasure.equivalent_width(starspectrum.wavelength.value, starspectrum.flux.value,
                                     specRange=[22468., 22471.5], continuum=cont34_fe1)  # Fe line
    fe2_ew = rvmeasure.equivalent_width(starspectrum.wavelength.value, starspectrum.flux.value, specRange=[22477.5, 22481.5],
                                     continuum=cont34_fe2)  # Fe line
    fe3_ew = rvmeasure.equivalent_width(starspectrum.wavelength.value, starspectrum.flux.value,
                                     specRange=[22498, 22502], continuum=cont34_fe3)  # Fe line


    k.write(str(removed_points[0])+'\t'+str(removed_points[1])+'\t'+str(removed_points[2])+'\t'+str(removed_points[3])+'\t'+str(si1_ew)+'\t'+str(cont35_si1)+'\t'+str(si2_ew)+'\t'+str(cont36_si2)+'\t'+str(s1_ew)+'\t'+str(cont34_s1)+'\t'+\
            str(fe1_ew)+'\t'+str(cont34_fe1)+'\t'+str(fe2_ew)+'\t'+str(cont34_fe2)+'\t'+str(fe3_ew)+'\t'+str(cont34_fe3)+'\n')

    #f = open('/u/rbentley/starkit_interpol_residuals/bosz_t'+str(removed_points[0])+'_logg'+str(removed_points[1])+'_mh'+str(removed_points[2])+'_a'+str(removed_points[3])+'.dat', 'w')

    #f.write('wavelength\tgrid flux\tstarkit flux\tresidual\n')
    #for j in range(len(res1)):
    #    f.write(str(w1[j])+'\t'+str(removed_flux[j])+'\t'+str(f1[j])+'\t'+str(res1[j])+'\n')
    #f.close()

k.close()
'''
for i in idx2:

    removed_points2 = points[i]
    removed_flux2 = values[i]

    print removed_points2
    print removed_flux2

    #grid.fluxes = np.delete(values, i, axis=0)
    #grid.index = np.delete(points, i, axis=0)

    print len(grid.index)


    starspectrum = Spectrum1D.from_array(dispersion=wavelengths, flux=removed_flux2,
                                           dispersion_unit=u.angstrom,
                                           uncertainty=removed_flux2*(1/100.))

    interp1 = Interpolate(starspectrum)
    norm1 = Normalize(starspectrum, 2)

    model = grid | interp1 | norm1

    setattr(model, 'teff_0', removed_points2[0])
    setattr(model, 'logg_0', removed_points2[1])
    setattr(model, 'mh_0', removed_points2[2])
    setattr(model, 'alpha_0', removed_points2[3])

    result2 = mtf.fit_array(starspectrum, model, R_fixed=25000.)

    for a in result2.median.keys():
        setattr(model, a, result2.median[a])

    print result2.median

    w2, f2 = model()

    res2 = starspectrum.flux.value - f2

plt.figure(figsize=(16, 12))


plt.plot(wavelengths, removed_flux2+0.6, label='BOSZ grid spectrum',
                     color='#000000', linewidth=5.0)

plt.plot(w2, f2+0.6, color='#33AFFF', linewidth=5.0)

plt.plot(w2, res2+0.2, label='Starkit interpolated model/residuals ([M/H]=-0.75)',
                    color='#33AFFF', linewidth=5.0)

plt.plot(wavelengths, removed_flux,
                     color='#000000', linewidth=5.0)

plt.plot(w1, f1, color='r', linewidth=5.0)

plt.plot(w1, res1, label='Starkit interpolated model/residuals ([M/H]=-1.25)',
                    color='r', linewidth=5.0)


plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
plt.axhline(y=-0.05, color='k', linestyle='--')

plt.axhline(y=0.25, color='k', linestyle='--')
plt.axhline(y=0.15, color='k', linestyle='--')

#plt.xlim(21500, 21800)

(linelocs, linelabels) = plotlines.extract_lines(angstrom=True, arcturus=True, molecules=False, wave_range=(21900, 22100))

alpha_line_residuals = []

fe_line_residuals = []

plot_idx = np.where((w1 > 21900) & (w1 < 22100))[0]

f1_on_plot = f1[plot_idx]

f2_on_plot = f2[plot_idx]

removed_flux1_plot = removed_flux[plot_idx]

removed_flux2_plot = removed_flux2[plot_idx]

for i in range(len(linelocs)):
    line = linelocs[i]
    if linelabels[i].replace('$', '') in ['O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Ti', 'C']:
        alpha_idx_w2 = int(np.where(w2 == (closest(w2, line)))[0])
        alpha_depth_w2 = np.amax(f2_on_plot) - f2[alpha_idx_w2]

        alpha_idx_w1 = int(np.where(w1 == (closest(w1, line)))[0])
        alpha_depth_w1 = np.amax(f1_on_plot) - f1[alpha_idx_w1]

        alpha_depth_bosz2 = np.amax(removed_flux2_plot) - removed_flux2[alpha_idx_w2]

        alpha_depth_bosz1 = np.amax(removed_flux1_plot) - removed_flux[alpha_idx_w1]

        alpha_line_residuals += [(linelabels[i].replace('$', '')+'\t'+str(line)+'\t'+str(alpha_depth_bosz2/alpha_depth_w2)+'\t'+str(alpha_depth_bosz1/alpha_depth_w1))]

    elif linelabels[i].replace('$', '') in ['Fe']:
        fe_idx_w2 = int(np.where(w2 == (closest(w2, line)))[0])
        fe_depth_w2 = np.amax(f2_on_plot) - f2[fe_idx_w2]

        fe_idx_w1 = int(np.where(w1 == (closest(w1, line)))[0])
        fe_depth_w1 = np.amax(f1_on_plot) - f1[fe_idx_w1]

        fe_depth_bosz2 = np.amax(removed_flux2_plot) - removed_flux2[fe_idx_w2]

        fe_depth_bosz1 = np.amax(removed_flux1_plot) - removed_flux[fe_idx_w1]

        fe_line_residuals += [(linelabels[i].replace('$', '')+'\t'+str(line)+'\t'+str(fe_depth_bosz2/fe_depth_w2)+'\t'+str(fe_depth_bosz1/fe_depth_w1))]

for i in alpha_line_residuals:
    print i

for i in fe_line_residuals:
    print i

#plt.ylim(-0.2, 1.3)


plt.legend(loc='upper left', fontsize=16)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux')
#plt.title('StarKit interpolation of PHOENIX grid points Teff='+str(removed_points[0])+', logg='+str(removed_points[1])+\
#            ', [M/H]='+str(removed_points[2])+', [alpha/Fe]='+str(removed_points[3]))
plt.title('StarKit spectrum of BOSZ grid points Teff='+str(removed_points[0])+', logg='+str(removed_points[1])+\
            ', [M/H]='+str(removed_points[2])+', [alpha/Fe]='+str(removed_points[3])+'\n'+\
            'and  Teff='+str(removed_points2[0])+', logg='+str(removed_points2[1])+\
            ', [M/H]='+str(removed_points2[2])+', [alpha/Fe]='+str(removed_points2[3])+'\n')
plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=15)
plt.show()
'''