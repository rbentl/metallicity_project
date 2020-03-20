import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
import math
from astropy import units as u
from astropy.modeling import models,fitting
from astropy.modeling import Model
from astropy.io import ascii
from astropy.io import fits
from astroquery.vizier import Vizier
import starkit
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from starkit.fix_spectrum1d import SKSpectrum1D
from specutils import read_fits_file,plotlines
import numpy as np
import os,scipy
from specutils import Spectrum1D,rvmeasure
import datetime,glob
import model_tester_updated as mtu
from matplotlib.backends.backend_pdf import PdfPages
import operator
import sys
from matplotlib.pyplot import cm
import multi_order_fitting_functions as mtf
from scipy.stats.stats import pearsonr, chisquare
from scipy.optimize import curve_fit
from scipy import ndimage as nd
import scipy.stats as stats


class Splitter3(Model):
    # split a single spectrum into 3
    inputs = ('w', 'f')
    outputs = ('w', 'f', 'w', 'f', 'w', 'f')

    def evaluate(self, w, f):
        return w, f, w, f, w, f


class Combiner3(Model):
    # combines the likelihood for four spectra
    inputs = ('l1', 'l2', 'l3')
    outputs = ('ltot',)

    def evaluate(self, l1, l2, l3):
        return l1 + l2 + l3

def linear_fit(x,m,b):
    return m*x + b

def constant_fit(x,c):
    return c

def connect_points(ax,x1,x2,y1,y2):
    for i in range(len(x1)):
        ax.plot([x1[i],x2[i]],[y1[i],y2[i]], color='k', alpha=0.4,linestyle='--')



def make_model_three_order(spectrum1,spectrum2,spectrum3,grid, convolve=None):

    if convolve is not None:
        r_val = convolve
    else:
        r_val = 24000

    interp1 = Interpolate(spectrum1)
    convolve1 = InstrumentConvolveGrating.from_grid(grid, R=r_val)
    rot1 = RotationalBroadening.from_grid(grid, vrot=np.array([10.0]))
    norm1 = Normalize(spectrum1, 2)

    interp2 = Interpolate(spectrum2)
    convolve2 = InstrumentConvolveGrating.from_grid(grid, R=r_val)
    norm2 = Normalize(spectrum2, 2)

    interp3 = Interpolate(spectrum3)
    convolve3 = InstrumentConvolveGrating.from_grid(grid, R=r_val)
    norm3 = Normalize(spectrum3, 2)


    model = grid | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
                 convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
                 norm1 & norm2 & norm3

    return model


def load_full_grid_phoenix():
    g = load_grid('/u/ghezgroup/data/metallicity/nirspec/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')
    return g

def load_full_grid_bosz():
    g = load_grid('/u/ghezgroup/data/metallicity/nirspec/grids/bosz_t3500_7000_w20000_24000_R50000.h5')
    return g

def make_fit_result_plots_k_band_three_order_unmasked_only(grids=None, include_phoenix=False, include_spec_text=False, plot_order=35):
    snr = 30.

    outputpath = 'plots_for_paper'

    result_title = 'BOSZ'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/' + outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/' + outputpath)

    cal_star_info_all = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))
    koa_star_info_all = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info_koa.dat', delimiter='\t', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[:-1]]
    koa_star_names = [x[0] for x in koa_star_info_all]

    textsize = 10
    textsize_spec = 7

    if plot_order==34:
        spec_min = 22540
        spec_max = 22700

    elif plot_order==36:
        spec_min = 21240
        spec_max = 21400

    else:
        spec_min = 21940
        spec_max = 22100

    bosz_vals = {'teff': [], \
                 'logg': [], \
                 'mh': [], \
                 'alpha': []}

    bosz_offsets = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    bosz_sigmas = {'teff': [], \
                   'logg': [], \
                   'mh': [], \
                   'alpha': []}

    phoenix_vals = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    phoenix_offsets = {'teff': [], \
                       'logg': [], \
                       'mh': [], \
                       'alpha': []}

    phoenix_sigmas = {'teff': [], \
                      'logg': [], \
                      'mh': [], \
                      'alpha': []}

    ap_values = {'teff': [], \
                 'logg': [], \
                 'mh': [], \
                 'alpha': []}

    koa_vals = {'teff': [], \
                'logg': [], \
                'mh': [], \
                'alpha': []}

    koa_offsets = {'teff': [], \
                   'logg': [], \
                   'mh': [], \
                   'alpha': []}

    koa_sigmas = {'teff': [], \
                  'logg': [], \
                  'mh': [], \
                  'alpha': []}

    koa_ap_values = {'teff': [], \
                     'logg': [], \
                     'mh': [], \
                     'alpha': []}

    koa_phoenix_vals = {'teff': [], \
                'logg': [], \
                'mh': [], \
                'alpha': []}

    koa_phoenix_offsets = {'teff': [], \
                   'logg': [], \
                   'mh': [], \
                   'alpha': []}

    koa_phoenix_sigmas = {'teff': [], \
                  'logg': [], \
                  'mh': [], \
                  'alpha': []}

    koa_phoenix_ap_values = {'teff': [], \
                     'logg': [], \
                     'mh': [], \
                     'alpha': []}


    chi2_vals = []
    koa_chi2_vals = []

    if grids is not None:
        phoenix = grids[1]
        bosz = grids[0]

    else:
        phoenix = load_full_grid_phoenix()

        bosz = load_full_grid_bosz()
        # bosz = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w15000_17000_R25000.h5')

    print cal_star_names
    for name in cal_star_names:
        star_ind = cal_star_names.index(name)
        cal_star_info = cal_star_info_all[star_ind]

        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + name + '_order34-36_bosz_adderr.h5')

        print bosz_result
        bosz_bounds = bosz_result.calculate_sigmas(1)
        '''
        bosz_result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1'])

        plt.savefig(
            '/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + name + '_' + result_title + '_corner.png')
        plt.clf()
        '''
        bosz_sigmas['teff'] += [np.sqrt(
            ((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2)]
        bosz_sigmas['logg'] += [np.sqrt(
            ((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[8] ** 2 + 0.1 ** 2)]
        bosz_sigmas['mh'] += [
            np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[6] ** 2 + 0.03 ** 2)]
        bosz_sigmas['alpha'] += [np.sqrt(
            ((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[9] ** 2 + 0.03 ** 2)]

        bosz_sigmas_one = [np.sqrt(
            ((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2), + \
                               np.sqrt(((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                                   8] ** 2 + 0.1 ** 2), + \
                               np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                                   6] ** 2 + 0.03 ** 2), + \
                               np.sqrt(
                                   ((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[
                                       9] ** 2 + 0.03 ** 2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)

        if include_phoenix:
            phoenix_result = MultiNestResult.from_hdf5(
                '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + name + '_order34-36_phoenix_R24000.0_adderr.h5')

            phoenix_bounds = phoenix_result.calculate_sigmas(1)
            '''
            phoenix_result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1'])

            plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + name + '_' + result_title + '_corner_phoenix.png')
            plt.clf()
            '''
            phoenix_sigmas['teff'] += [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[
                    7] ** 2 + 91.5 ** 2)]  # cal_star_info[7]
            phoenix_sigmas['logg'] += [np.sqrt(
                ((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                    8] ** 2 + 0.11 ** 2)]  # cal_star_info[8]
            phoenix_sigmas['mh'] += [np.sqrt(
                ((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                    6] ** 2 + 0.05 ** 2)]  # cal_star_info[6]
            phoenix_sigmas['alpha'] += [np.sqrt(
                ((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[
                    9] ** 2 + 0.05 ** 2)]  # cal_star_info[9]

            phoenix_sigmas_one = [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[
                    7] ** 2 + 91.5 ** 2), + \
                                      np.sqrt(((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 +
                                              cal_star_info[8] ** 2 + 0.1 ** 2), + \
                                      np.sqrt(((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 +
                                              cal_star_info[6] ** 2 + 0.03 ** 2), + \
                                      np.sqrt(((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 +
                                              cal_star_info[9] ** 2 + 0.03 ** 2)]

            phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)

            phoenix_vals['teff'] += [phoenix_result.median['teff_0']]
            phoenix_vals['logg'] += [phoenix_result.median['logg_0']]
            phoenix_vals['mh'] += [phoenix_result.median['mh_0']]
            phoenix_vals['alpha'] += [phoenix_result.median['alpha_0']]

            phoenix_offsets['teff'] += [phoenix_result.median['teff_0'] - cal_star_info[2]]
            phoenix_offsets['logg'] += [phoenix_result.median['logg_0'] - cal_star_info[3]]
            phoenix_offsets['mh'] += [phoenix_result.median['mh_0'] - cal_star_info[1]]
            phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0'] - cal_star_info[4]]

        bosz_vals['teff'] += [bosz_result.median['teff_0']]
        bosz_vals['logg'] += [bosz_result.median['logg_0']]
        bosz_vals['mh'] += [bosz_result.median['mh_0']]
        bosz_vals['alpha'] += [bosz_result.median['alpha_0']]

        bosz_offsets['teff'] += [bosz_result.median['teff_0'] - cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0'] - cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0'] - cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0'] - cal_star_info[4]]

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        bmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

        if include_phoenix:
            pmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, phoenix)

            for a in phoenix_result.median.keys():
                setattr(pmodel, a, phoenix_result.median[a])

            pw1, pf1, pw2, pf2, pw3, pf3 = pmodel()
            phoenix_res1 = starspectrum34.flux.value - pf1
            phoenix_res2 = starspectrum35.flux.value - pf2
            phoenix_res3 = starspectrum36.flux.value - pf3

            phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 / (len(phoenix_res1) + len(phoenix_res2) + len(phoenix_res3))

            phoenix_deltas = np.around(
                [phoenix_result.median['teff_0'] - cal_star_info[2], phoenix_result.median['logg_0'] - cal_star_info[3],
                 phoenix_result.median['mh_0'] - cal_star_info[1], phoenix_result.median['alpha_0'] - cal_star_info[4]],
                decimals=2)

        else:
            phoenix_chi2 = 0.0

        amodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1, bw2, bf2, bw3, bf3 = bmodel()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1) + len(bosz_res2) + len(bosz_res3))

        bosz_deltas = np.around(
            [bosz_result.median['teff_0'] - cal_star_info[2], bosz_result.median['logg_0'] - cal_star_info[3],
             bosz_result.median['mh_0'] - cal_star_info[1], bosz_result.median['alpha_0'] - cal_star_info[4]],
            decimals=2)

        setattr(amodel, 'teff_0', cal_star_info[2])
        setattr(amodel, 'logg_0', cal_star_info[3])
        setattr(amodel, 'mh_0', cal_star_info[1])
        setattr(amodel, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = amodel()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))

        chi2_vals += [
            (np.round_(bosz_chi2, decimals=2), np.round_(phoenix_chi2, decimals=2), np.round_(apogee_chi2, decimals=2))]

        plt.figure(figsize=(4.25, 3.5))

        props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

        if include_spec_text:
            plt.text(22090, 0.65,
                    '$T_{eff}:$' + str(cal_star_info[2]) + '$\pm$' + str(cal_star_info[7]) + '\n$log g:$' + str(
                        cal_star_info[3]) + '$\pm$' + str(cal_star_info[8]) + \
                    '\n$[M/H]:$' + str(cal_star_info[1]) + '$\pm$' + str(cal_star_info[6]) + '\n' + r'$\alpha$:' + str(
                        cal_star_info[4]) + '$\pm$' + str(cal_star_info[9]), fontsize=textsize_spec, bbox=props)

            plt.text(22090, 0.4, 'BOSZ fit offsets:\n$\Delta T_{eff}:$' + str(bosz_deltas[0]) + '$\pm$' + str(
                bosz_sigmas_one[0]) + '\n$\Delta log g:$' + str(bosz_deltas[1]) + '$\pm$' + str(bosz_sigmas_one[1]) + \
                    '\n$\Delta [M/H]:$' + str(bosz_deltas[2]) + '$\pm$' + str(
                bosz_sigmas_one[2]) + '\n' + r'$\Delta$ [$\alpha$/Fe]:' + str(bosz_deltas[3]) + '$\pm$' + str(
                bosz_sigmas_one[3]), fontsize=textsize_spec, bbox=props)

            if include_phoenix:
                plt.text(22090, 0.1, 'PHOENIX fit offsets:\n$\Delta T_{eff}:$' + str(phoenix_deltas[0]) + '$\pm$' + str(
                    phoenix_sigmas_one[0]) + '\n$\Delta log g:$' + str(phoenix_deltas[1]) + '$\pm$' + str(
                    phoenix_sigmas_one[1]) + \
                        '\n$\Delta [M/H]:$' + str(phoenix_deltas[2]) + '$\pm$' + str(
                    phoenix_sigmas_one[2]) + '\n' + r'$\Delta$$\alpha$:' + str(phoenix_deltas[3]) + '$\pm$' + str(
                    phoenix_sigmas_one[3]), fontsize=textsize_spec, bbox=props)

        plt.plot(starspectrum34.wavelength.value / (bosz_result.median['vrad_3'] / 3e5 + 1.0),
                 starspectrum34.flux.value,
                 color='#000000', label='Calibrator: '+name, linewidth=1.0)
        plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0),
                 starspectrum35.flux.value,
                 color='#000000', linewidth=1.0)
        plt.plot(starspectrum36.wavelength.value / (bosz_result.median['vrad_5'] / 3e5 + 1.0),
                 starspectrum36.flux.value,
                 color='#000000', linewidth=1.0)

        plt.plot(bw1 / (bosz_result.median['vrad_3'] / 3e5 + 1.0), bf1, color='#33AFFF',
                 linewidth=1.0)
        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bf2, color='#33AFFF', label='BOSZ Model/Residuals',
                 linewidth=1.0)
        plt.plot(bw3 / (bosz_result.median['vrad_5'] / 3e5 + 1.0), bf3, color='#33AFFF',
                 linewidth=1.0)

        plt.plot(bw1 / (bosz_result.median['vrad_3'] / 3e5 + 1.0), bosz_res1, color='#33AFFF', linewidth=1.0)
        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bosz_res2, color='#33AFFF', linewidth=1.0)
        plt.plot(bw3 / (bosz_result.median['vrad_5'] / 3e5 + 1.0), bosz_res3, color='#33AFFF', linewidth=1.0)

        plt.xticks(np.arange(spec_min, spec_max, step=30))
        plt.tick_params(axis='y', which='major', labelsize=textsize_spec)
        plt.tick_params(axis='x', which='major', labelsize=textsize_spec)

        if include_phoenix:
            plt.plot(pw1 / (phoenix_result.median['vrad_3'] / 3e5 + 1.0), pf1, color='#FEBE4E', linewidth=1.0)
            plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), pf2, color='#FEBE4E',
                     label='PHOENIX Model/Residuals', linewidth=1.0)
            plt.plot(pw3 / (phoenix_result.median['vrad_5'] / 3e5 + 1.0), pf3, color='#FEBE4E', linewidth=1.0)

            plt.plot(pw1 / (phoenix_result.median['vrad_3'] / 3e5 + 1.0), phoenix_res1, color='#FEBE4E', linewidth=1.0)
            plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), phoenix_res2, color='#FEBE4E', linewidth=1.0)
            plt.plot(pw3 / (phoenix_result.median['vrad_5'] / 3e5 + 1.0), phoenix_res3, color='#FEBE4E', linewidth=1.0)


        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%', linewidth=1.0)
        plt.axhline(y=-0.05, color='k', linestyle='--', linewidth=1.0)


        plt.xlim(spec_min, spec_max)
        plt.ylim(-0.2, 1.3)

        plt.legend(loc='center left', fontsize=textsize_spec)
        plt.xlabel('Wavelength (Angstroms)', fontsize=textsize_spec, labelpad=0)
        plt.ylabel('Normalized Flux', fontsize=textsize_spec, labelpad=0)
        #plt.title(result_title + ' fits and residuals for ' + starname)
        plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=textsize_spec)

        if include_phoenix:
            plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/plots_for_paper/specs_corners/' + name + '_' + result_title + '_spectrum_o'+str(plot_order)+'_phoenix.pdf', bbox_inches='tight')
        else:
            plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/plots_for_paper/specs_corners/' + name + '_' + result_title + '_spectrum_o'+str(plot_order)+'.pdf', bbox_inches='tight')

        plt.clf()

        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]

    for starname in koa_star_names:

        star_ind = koa_star_names.index(starname)
        cal_star_info = koa_star_info_all[star_ind]

        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + starname + '_order34-36_bosz_fit_R_adderr.h5')

        bosz_bounds = bosz_result.calculate_sigmas(1)

        koa_vals['teff'] += [bosz_result.median['teff_0']]
        koa_vals['logg'] += [bosz_result.median['logg_0']]
        koa_vals['mh'] += [bosz_result.median['mh_0']]
        koa_vals['alpha'] += [bosz_result.median['alpha_0']]

        if include_phoenix:
            phoenix_result = MultiNestResult.from_hdf5(
                '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + starname + '_order34-36_phoenix_fit_R_adderr.h5')

            phoenix_bounds = phoenix_result.calculate_sigmas(1)
            '''
            phoenix_result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1'])

            plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + name + '_' + result_title + '_corner_phoenix.png')
            plt.clf()
            '''
            koa_phoenix_sigmas['teff'] += [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[
                    7] ** 2 + 91.5 ** 2)]  # cal_star_info[7]
            koa_phoenix_sigmas['logg'] += [np.sqrt(
                ((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                    8] ** 2 + 0.11 ** 2)]  # cal_star_info[8]
            koa_phoenix_sigmas['mh'] += [np.sqrt(
                ((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                    6] ** 2 + 0.05 ** 2)]  # cal_star_info[6]
            koa_phoenix_sigmas['alpha'] += [np.sqrt(
                ((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[
                    9] ** 2 + 0.05 ** 2)]  # cal_star_info[9]

            phoenix_sigmas_one = [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[
                    7] ** 2 + 91.5 ** 2), + \
                                      np.sqrt(((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 +
                                              cal_star_info[8] ** 2 + 0.1 ** 2), + \
                                      np.sqrt(((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 +
                                              cal_star_info[6] ** 2 + 0.03 ** 2), + \
                                      np.sqrt(((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 +
                                              cal_star_info[9] ** 2 + 0.03 ** 2)]

            phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)

            koa_phoenix_vals['teff'] += [phoenix_result.median['teff_0']]
            koa_phoenix_vals['logg'] += [phoenix_result.median['logg_0']]
            koa_phoenix_vals['mh'] += [phoenix_result.median['mh_0']]
            koa_phoenix_vals['alpha'] += [phoenix_result.median['alpha_0']]

            koa_phoenix_offsets['teff'] += [phoenix_result.median['teff_0'] - cal_star_info[2]]
            koa_phoenix_offsets['logg'] += [phoenix_result.median['logg_0'] - cal_star_info[3]]
            koa_phoenix_offsets['mh'] += [phoenix_result.median['mh_0'] - cal_star_info[1]]
            koa_phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0'] - cal_star_info[4]]

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/koa_specs/' + starname + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/koa_specs/' + starname + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/koa_specs/' + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        # model = make_model_three_order(starspectrum34,starspectrum35,starspectrum36, g)
        interp1 = Interpolate(starspectrum34)
        convolve1 = InstrumentConvolveGrating.from_grid(bosz, R=bosz_result.median['R_6'])  # R=24000
        # convolve1 = starkit.base.operations.spectrograph.InstrumentDeltaLambdaConstant.from_grid(g, delta_lambda=0.961)
        rot1 = RotationalBroadening.from_grid(bosz, vrot=np.array([10.0]))
        norm1 = Normalize(starspectrum34, 2)

        interp2 = Interpolate(starspectrum35)
        convolve2 = InstrumentConvolveGrating.from_grid(bosz, R=bosz_result.median['R_7'])
        # convolve2 = starkit.base.operations.spectrograph.InstrumentDeltaLambdaConstant.from_grid(g, delta_lambda=0.961)
        norm2 = Normalize(starspectrum35, 2)

        interp3 = Interpolate(starspectrum36)
        convolve3 = InstrumentConvolveGrating.from_grid(bosz, R=bosz_result.median['R_8'])
        # convolve3 = starkit.base.operations.spectrograph.InstrumentDeltaLambdaConstant.from_grid(g, delta_lambda=0.961)
        norm3 = Normalize(starspectrum36, 2)

        model = bosz | rot1 | Splitter3() | DopplerShift(vrad=0) & DopplerShift(vrad=0) & DopplerShift(vrad=0) | \
                convolve1 & convolve2 & convolve3 | interp1 & interp2 & interp3 | \
                norm1 & norm2 & norm3

        for a in bosz_result.median.keys():
            setattr(model, a, bosz_result.median[a])

        print bosz_result

        bw1, bf1, bw2, bf2, bw3, bf3 = model()

        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1) + len(bosz_res2) + len(bosz_res3))

        bosz_deltas = [bosz_result.median['teff_0'] - cal_star_info[2], bosz_result.median['logg_0'] - cal_star_info[3],
                  bosz_result.median['mh_0'] - cal_star_info[1], bosz_result.median['alpha_0'] - cal_star_info[4]]

        bosz_deltas = np.around(bosz_deltas, decimals=2)

        koa_offsets['teff'] += [bosz_deltas[0]]
        koa_offsets['logg'] += [bosz_deltas[1]]
        koa_offsets['mh'] += [bosz_deltas[2]]
        koa_offsets['alpha'] += [bosz_deltas[3]]

        koa_sigmas['teff'] += [(bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2]
        koa_sigmas['logg'] += [(bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2]
        koa_sigmas['mh'] += [(bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2]
        koa_sigmas['alpha'] += [(bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2]

        bosz_sigmas_one = [np.sqrt(((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2), + \
                               np.sqrt(((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[8] ** 2 + 0.1 ** 2), + \
                               np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[6] ** 2 + 0.03 ** 2), + \
                               np.sqrt(((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[9] ** 2 + 0.03 ** 2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)

        if include_phoenix:
            pmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, phoenix)

            print phoenix_result

            for a in phoenix_result.median.keys():
                setattr(pmodel, a, phoenix_result.median[a])

            pw1, pf1, pw2, pf2, pw3, pf3 = pmodel()
            phoenix_res1 = starspectrum34.flux.value - pf1
            phoenix_res2 = starspectrum35.flux.value - pf2
            phoenix_res3 = starspectrum36.flux.value - pf3

            phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 / (len(phoenix_res1) + len(phoenix_res2) + len(phoenix_res3))

            phoenix_deltas = np.around(
                [phoenix_result.median['teff_0'] - cal_star_info[2], phoenix_result.median['logg_0'] - cal_star_info[3],
                 phoenix_result.median['mh_0'] - cal_star_info[1], phoenix_result.median['alpha_0'] - cal_star_info[4]],
                decimals=2)

            phoenix_sigmas_one = [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2), + \
                                   np.sqrt(
                                       ((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                                           8] ** 2 + 0.1 ** 2), + \
                                   np.sqrt(((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                                       6] ** 2 + 0.03 ** 2), + \
                                   np.sqrt(((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 +
                                           cal_star_info[9] ** 2 + 0.03 ** 2)]
            phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)


        w1, f1, w2, f2, w3, f3 = model()
        res1 = starspectrum34.flux.value - f1
        res2 = starspectrum35.flux.value - f2
        res3 = starspectrum36.flux.value - f3

        chi2 = np.sum((res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        chi2 = chi2 + np.sum((res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        chi2 = chi2 + np.sum((res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        chi2 = chi2 / (len(res1) + len(res2) + len(res3))

        setattr(bmodel, 'teff_0', cal_star_info[2])
        setattr(bmodel, 'logg_0', cal_star_info[3])
        setattr(bmodel, 'mh_0', cal_star_info[1])
        setattr(bmodel, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = bmodel()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))

        koa_chi2_vals += [(chi2, bosz_chi2, apogee_chi2)]
        '''
        bosz_result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1'])

        plt.savefig(
            '/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + starname + '_' + result_title + '_corner.png')
        plt.clf()
        '''

        plt.figure(figsize=(4.25, 3.5))

        props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

        if include_spec_text:
            plt.text(22090, 0.65,
                    '$T_{eff}:$' + str(cal_star_info[2]) + '$\pm$' + str(cal_star_info[7]) + '\n$log g:$' + str(
                        cal_star_info[3]) + '$\pm$' + str(cal_star_info[8]) + \
                    '\n$[M/H]:$' + str(cal_star_info[1]) + '$\pm$' + str(cal_star_info[6]) + '\n' + r'$\alpha$:' + str(
                        cal_star_info[4]) + '$\pm$' + str(cal_star_info[9]), fontsize=textsize_spec, bbox=props)

            plt.text(22090, 0.4, 'BOSZ fit offsets:\n$\Delta T_{eff}:$' + str(bosz_deltas[0]) + '$\pm$' + str(
                bosz_sigmas_one[0]) + '\n$\Delta log g:$' + str(bosz_deltas[1]) + '$\pm$' + str(bosz_sigmas_one[1]) + \
                    '\n$\Delta [M/H]:$' + str(bosz_deltas[2]) + '$\pm$' + str(
                bosz_sigmas_one[2]) + '\n' + r'$\Delta$ [$\alpha$/Fe]:' + str(bosz_deltas[3]) + '$\pm$' + str(
                bosz_sigmas_one[3]), fontsize=textsize_spec, bbox=props)

            if include_phoenix:
                plt.text(22090, 0.1, 'PHOENIX fit offsets:\n$\Delta T_{eff}:$' + str(phoenix_deltas[0]) + '$\pm$' + str(
                    phoenix_sigmas_one[0]) + '\n$\Delta log g:$' + str(phoenix_deltas[1]) + '$\pm$' + str(
                    phoenix_sigmas_one[1]) + \
                        '\n$\Delta [M/H]:$' + str(phoenix_deltas[2]) + '$\pm$' + str(
                    phoenix_sigmas_one[2]) + '\n' + r'$\Delta$$\alpha$:' + str(phoenix_deltas[3]) + '$\pm$' + str(
                    phoenix_sigmas_one[3]), fontsize=textsize_spec, bbox=props)

        plt.plot(starspectrum34.wavelength.value / (bosz_result.median['vrad_3'] / 3e5 + 1.0),
                 starspectrum34.flux.value,
                 color='#000000', label='Calibrator: '+starname, linewidth=1.0)
        plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0),
                 starspectrum35.flux.value,
                 color='#000000', linewidth=1.0)
        plt.plot(starspectrum36.wavelength.value / (bosz_result.median['vrad_5'] / 3e5 + 1.0),
                 starspectrum36.flux.value,
                 color='#000000', linewidth=1.0)

        plt.plot(bw1 / (bosz_result.median['vrad_3'] / 3e5 + 1.0), bf1, color='#33AFFF',
                 linewidth=1.0)
        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bf2, color='#33AFFF', label='BOSZ Model/Residuals',
                 linewidth=1.0)
        plt.plot(bw3 / (bosz_result.median['vrad_5'] / 3e5 + 1.0), bf3, color='#33AFFF',
                 linewidth=1.0)

        plt.plot(bw1 / (bosz_result.median['vrad_3'] / 3e5 + 1.0), bosz_res1, color='#33AFFF', linewidth=1.0)
        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bosz_res2, color='#33AFFF', linewidth=1.0)
        plt.plot(bw3 / (bosz_result.median['vrad_5'] / 3e5 + 1.0), bosz_res3, color='#33AFFF', linewidth=1.0)

        if include_phoenix:
            plt.plot(pw1 / (phoenix_result.median['vrad_3'] / 3e5 + 1.0), pf1, color='#FEBE4E', linewidth=1.0)
            plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), pf2, color='#FEBE4E',
                     label='PHOENIX Model/Residuals', linewidth=1.0)
            plt.plot(pw3 / (phoenix_result.median['vrad_5'] / 3e5 + 1.0), pf3, color='#FEBE4E', linewidth=1.0)

            plt.plot(pw1 / (phoenix_result.median['vrad_3'] / 3e5 + 1.0), phoenix_res1, color='#FEBE4E', linewidth=1.0)
            plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), phoenix_res2, color='#FEBE4E', linewidth=1.0)
            plt.plot(pw3 / (phoenix_result.median['vrad_5'] / 3e5 + 1.0), phoenix_res3, color='#FEBE4E', linewidth=1.0)


        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%', linewidth=1.0)
        plt.axhline(y=-0.05, color='k', linestyle='--', linewidth=1.0)

        plt.xticks(np.arange(spec_min, spec_max, step=30))
        plt.tick_params(axis='y', which='major', labelsize=textsize_spec)
        plt.tick_params(axis='x', which='major', labelsize=textsize_spec)

        plt.xlim(spec_min, spec_max)
        plt.ylim(-0.2, 1.3)

        plt.legend(loc='center left', fontsize=textsize_spec)
        plt.xlabel('Wavelength (Angstroms)', fontsize=textsize_spec, labelpad=0)
        plt.ylabel('Normalized Flux', fontsize=textsize_spec, labelpad=0)
        #plt.title(result_title + ' fits and residuals for ' + starname)
        plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=textsize_spec)

        if include_phoenix:
            plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/plots_for_paper/specs_corners/' + starname + '_' + result_title + '_spectrum_o'+str(plot_order)+'_phoenix.pdf', bbox_inches='tight')
        else:
            plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/plots_for_paper/specs_corners/' + starname + '_' + result_title + '_spectrum_o'+str(plot_order)+'.pdf', bbox_inches='tight')

        plt.clf()

        koa_ap_values['teff'] += [cal_star_info[2]]
        koa_ap_values['logg'] += [cal_star_info[3]]
        koa_ap_values['mh'] += [cal_star_info[1]]
        koa_ap_values['alpha'] += [cal_star_info[4]]

    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys()) + 1, ncols=1, figsize=(7.5, 9))

    fig.subplots_adjust(hspace=0.025, wspace=0.0)

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    x_axis_b = ap_values['mh']
    x_axis_p = ap_values['mh']
    koa_x_axis_b = koa_ap_values['mh']
    koa_x_axis_p = koa_ap_values['mh']

    x_axis_b = np.array([float(i) for i in x_axis_b])
    x_axis_p = np.array([float(i) for i in x_axis_p])
    koa_x_axis_b = np.array([float(i) for i in koa_x_axis_b])
    koa_x_axis_p = np.array([float(i) for i in koa_x_axis_p])


    for i in range(len(ax) - 1):
        combined_offsets = np.array(bosz_offsets[bosz_offsets.keys()[i]] + koa_offsets[koa_offsets.keys()[i]])
        combined_sigmas = np.array(bosz_sigmas[bosz_offsets.keys()[i]] + koa_sigmas[koa_offsets.keys()[i]])

        all_x_axis_b = np.concatenate((x_axis_b, koa_x_axis_b))

        b_stdev = np.std(combined_offsets)
        b_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + b_stdev ** 2) for x in combined_sigmas])
        b_offset_mean = np.average(combined_offsets, weights=b_sigmas_stdev ** -2)

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[bosz_offsets.keys()[i]],
                       color='#3349FF', marker='.', label='StarKit offset', ls='none', markersize=16)

        ax[i].errorbar(koa_x_axis_b, koa_offsets[koa_offsets.keys()[i]], yerr=koa_sigmas[koa_offsets.keys()[i]],
                       color='#3349FF', marker='d', label='StarKit offset (KOA spectrum)', ls='none',
                       markersize=8)



        if include_phoenix:
            combined_offsets_p = np.array(phoenix_offsets[phoenix_offsets.keys()[i]] + koa_phoenix_offsets[koa_phoenix_offsets.keys()[i]])
            combined_sigmas_p = np.array(phoenix_sigmas[phoenix_offsets.keys()[i]] + koa_phoenix_sigmas[koa_phoenix_offsets.keys()[i]])

            all_x_axis_p = np.concatenate((x_axis_p, koa_x_axis_p))

            p_stdev = np.std(combined_offsets_p)
            p_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + p_stdev ** 2) for x in combined_sigmas_p])
            p_offset_mean = np.average(combined_offsets_p, weights=p_sigmas_stdev ** -2)

            print phoenix_sigmas[phoenix_offsets.keys()[i]]

            ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],
                           color='#FEBE4E', marker='.', ls='none', markersize=16)

            ax[i].errorbar(koa_x_axis_p, koa_phoenix_offsets[koa_offsets.keys()[i]], yerr=koa_phoenix_sigmas[koa_offsets.keys()[i]],
                           color='#FEBE4E', marker='d', ls='none',
                           markersize=8)

            connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[bosz_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])

            connect_points(ax[i],koa_x_axis_b,koa_x_axis_p, koa_offsets[koa_offsets.keys()[i]], koa_phoenix_offsets[koa_offsets.keys()[i]])

        bpopt, bpcov = curve_fit(linear_fit, all_x_axis_b, combined_offsets, sigma=combined_sigmas)

        bpopt_c, bpcov_c = curve_fit(constant_fit, all_x_axis_b, combined_offsets, sigma=combined_sigmas)

        fit_res_b = np.array(combined_offsets - linear_fit(all_x_axis_b, bpopt[0], bpopt[1]))
        chi2_b_fit = np.sum((fit_res_b) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_red = chi2_b_fit / (len(fit_res_b) - len(bpopt))

        fit_res_b_c = np.array(combined_offsets - constant_fit(all_x_axis_b, bpopt_c[0]))
        chi2_b_fit_c = np.sum((fit_res_b_c) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_c_red = chi2_b_fit_c / (len(fit_res_b_c) - len(bpopt_c))

        f_b = (chi2_b_fit_c - chi2_b_fit)
        f_b = f_b / (len(bpopt) - len(bpopt_c))
        f_b = f_b * (len(fit_res_b) - len(bpopt)) / chi2_b_fit

        pval = stats.f.sf(f_b, 2, 1)

        ax[i].axhline(y=b_offset_mean, color='#3349FF', linestyle='--', label='Mean offset')
        ax[i].axhspan(b_offset_mean - b_stdev, b_offset_mean + b_stdev,
                      color='#3349FF', alpha=0.2)

        if include_phoenix:
            ax[i].axhline(y=p_offset_mean, color='#FEBE4E', linestyle='--')
            ax[i].axhspan(p_offset_mean - p_stdev, p_offset_mean + p_stdev,
                          color='#FEBE4E', alpha=0.2)

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()

        ytext = ymax / 2.

        if include_phoenix:
            ax[i].text((xmax-xmin)*0.97+xmin, (ymax-ymin)*0.4+ymin, 'BOSZ $\sigma$:' + str(np.round_(b_stdev, decimals=2)) + '\nBOSZ Mean:' +
                       str(np.round_(b_offset_mean, decimals=2)) + '\nPHOENIX $\sigma$:' + str(np.round_(p_stdev, decimals=2)) +
                        '\nPHOENIX Mean:' + str(np.round_(p_offset_mean, decimals=2)), fontsize=10,
                       bbox=props)

        else:
            ax[i].text((xmax-xmin)*0.97+xmin, (ymax-ymin)*0.4+ymin, 'BOSZ $\sigma$:' + str(np.round_(b_stdev, decimals=2)) + '\nBOSZ Mean:' + str(
                np.round_(b_offset_mean, decimals=2)), fontsize=10,
                bbox=props)  # +'\np-value: '+str(np.round_(pval,decimals=2), +'\nF-statistic: '+str(np.round_(f_b,decimals=2))+'\n5% crit val: '+str(np.round_(18.513,decimals=2))



        ax[i].axhline(y=0., color='k', linestyle='--')

        ax[i].tick_params(axis='y', which='major', labelsize=10)
        ax[i].tick_params(axis='x', which='major', labelsize=1)

    ax[-1].tick_params(axis='both', which='major', labelsize=10)

    ax[-1].plot(x_axis_b, [x[0] for x in chi2_vals], color='#3349FF', marker='.', ls='none', markersize=16)
    ax[-1].plot(koa_x_axis_b, [x[0] for x in koa_chi2_vals], color='#3349FF', marker='d', ls='none', markersize=9)
    ax[-1].plot(x_axis_b, [x[2] for x in chi2_vals], color='r', marker='.', label='BOSZ grid at APOGEE values',
                ls='none', markersize=16)
    ax[-1].plot(koa_x_axis_b, [x[2] for x in koa_chi2_vals], color='r', marker='d', ls='none', markersize=9)

    connect_points(ax[-1], x_axis_b, x_axis_p, [x[0] for x in chi2_vals],[x[2] for x in chi2_vals])
    connect_points(ax[-1], koa_x_axis_b, koa_x_axis_p, [x[0] for x in koa_chi2_vals],[x[2] for x in koa_chi2_vals])

    if include_phoenix:
        ax[-1].plot(x_axis_p, [x[1] for x in chi2_vals], color='#FEBE4E', marker='.', ls='none', markersize=16)
        ax[-1].plot(koa_x_axis_p, [x[1] for x in koa_chi2_vals], color='#FEBE4E', marker='d', ls='none', markersize=9)

    # ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax) - 1].set_xlabel('APOGEE [M/H]', fontsize=11)

    ax[0].set_ylabel('log g offset', fontsize=11)
    ax[1].set_ylabel(r'[$\alpha$/Fe]' + ' offset', fontsize=11)
    ax[2].set_ylabel('$T_{eff}$ offset (K)', fontsize=11)
    ax[3].set_ylabel('[M/H] offset', fontsize=11)
    # ax[4].set_ylabel(r'[$\alpha$/H]'+' offset', fontsize=16)
    # ax[4].set_ylim(-0.7,0.7)
    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=11)

    if include_phoenix:
        ax[0].set_ylim(-3.5, 3.5)
        ax[1].set_ylim(-0.7, 0.7)
        ax[2].set_ylim(-1100, 1100)
        ax[3].set_ylim(-0.7, 0.7)

    else:
        ax[0].set_ylim(-1.5, 1.5)
        ax[1].set_ylim(-0.5, 0.5)
        ax[2].set_ylim(-450, 450)
        ax[3].set_ylim(-0.5, 0.5)

    ax[0].legend(fontsize=10, loc='upper left')
    ax[-1].legend(fontsize=10, loc='upper left')
    #ax[0].set_title('StarKit-APOGEE fit offsets with KOA spectra')
    if include_phoenix:
        plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/grid_offsets_bosz_phoenix.pdf',
            bbox_inches='tight')
    else:
        plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/grid_offsets_bosz.pdf',
                        bbox_inches='tight')

def make_fit_result_plots_spex(grids=None):

    snr = 30.

    outputpath = 'spex'

    result_title = 'BOSZ'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/'+outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/'+outputpath)

    cal_star_info_all = list(scipy.genfromtxt('/u/rbentley/metallicity/irtf_spex_stars.dat', delimiter=',', skip_header=1, dtype=None))
    cal_star_info_all.sort(key=lambda x: x[0])
    cal_star_info_all = filter(lambda x: np.isfinite(x[8]) and np.isfinite(x[10]) and np.isfinite(x[11]), cal_star_info_all)
    cal_star_names = [x[3] for x in cal_star_info_all]

    for i in cal_star_info_all:
        print i
    '''
    bosz_vals = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    bosz_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    ap_values = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}
    '''
    bosz_vals = {'teff': [], \
                    'logg': [], \
                    'mh': []}

    bosz_offsets = {'teff':[],\
               'logg':[],\
               'mh':[]}

    bosz_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[]}

    phoenix_offsets = {'teff':[],\
               'logg':[],\
               'mh':[]}

    ap_values = {'teff':[],\
               'logg':[],\
               'mh':[]}

    chi2_vals = []
    koa_chi2_vals = []

    if grids is not None:
        bosz = grids

    else:
        bosz = load_full_grid_bosz()

    print cal_star_names
    for name in cal_star_names:

        star_ind = cal_star_names.index(name)
        cal_star_info = cal_star_info_all[star_ind]
        if 'HD' in name:
            name_comps = name.split(' ')
            name = name_comps[0] + str(int(name_comps[1]))

        else:
            name = name.replace(' ','_')
        print name, cal_star_info
        file_path = glob.glob(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/spex/*' + name + '_bosz_spex_fit_R_adderr.h5')

        if not file_path:
            continue

        bosz_result = MultiNestResult.from_hdf5(file_path[0])
        #phoenix_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/PHOENIX_fits/sl_masked/mh_masked_sl_cutoff_0.0_' + name + '_order34-36_phoenix_adderr.h5')

        bosz_bounds = bosz_result.calculate_sigmas(1)

        #bosz_result.plot_triangle(parameters=['teff_0', 'logg_0', 'mh_0', 'alpha_0', 'vrot_1'])

        #plt.savefig('/u/rbentley/localcompute/fitting_plots/'+outputpath+'/'+name+'_'+result_title+'_corner.png')
        #plt.clf()


        bosz_sigmas['teff'] += [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + 0.**2)]
        bosz_sigmas['logg'] += [np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + 0.**2)]
        bosz_sigmas['mh'] += [np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + 0.**2)]
        #bosz_sigmas['alpha'] += [np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + 0.**2)]

        bosz_sigmas_one = [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + 0.**2), +\
                           np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + 0.**2), +\
                           np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + 0.**2),+\
                           np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + 0.**2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)

        #phoenix_bounds = phoenix_result.calculate_sigmas(1)

        #phoenix_sigmas['teff'] += [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + 91.5**2)] #cal_star_info[7]
        #phoenix_sigmas['logg'] += [np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + 0.11**2)] #cal_star_info[8]
        #phoenix_sigmas['mh'] += [np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + 0.05**2)] #cal_star_info[6]
        #phoenix_sigmas['alpha'] += [np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + 0.05**2)] #cal_star_info[9]

        #phoenix_sigmas_one = [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
        #                   np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
        #                   np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
        #                   np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        #phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)

        bosz_vals['teff'] += [bosz_result.median['teff_0']]
        bosz_vals['logg'] += [bosz_result.median['logg_0']]
        bosz_vals['mh'] += [bosz_result.median['mh_0']]
        #bosz_vals['alpha'] += [bosz_result.median['alpha_0']]

        bosz_offsets['teff'] += [bosz_result.median['teff_0']-cal_star_info[8]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0']-cal_star_info[10]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0']-cal_star_info[11]]
        #bosz_offsets['alpha'] += [bosz_result.median['alpha_0']-0.]



        file1 = '/u/tdo/research/metallicity/standards/*' + name + '.fits'

        spec1 = glob.glob(file1)[0]

        starspectrum = mtf.load_spex_spectra(spec1,waverange=[21000.,22910.],normalize=True)

        interp1 = Interpolate(starspectrum)
        convolve1 = InstrumentConvolveGrating.from_grid(bosz, R=2000.)
        rot1 = RotationalBroadening.from_grid(bosz, vrot=np.array([10.0]))
        norm1 = Normalize(starspectrum, 2)

        bmodel = bosz | rot1 | DopplerShift(vrad=0) | convolve1 | interp1 | norm1

        amodel = bmodel

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1 = bmodel()
        bosz_res1 = starspectrum.flux.value - bf1

        print bosz_res1, len(bosz_res1)

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / len(bosz_res1)

        bosz_deltas = np.around([bosz_result.median['teff_0'] - cal_star_info[8], bosz_result.median['logg_0'] - cal_star_info[10],
                  bosz_result.median['mh_0'] - cal_star_info[11], bosz_result.median['alpha_0'] - 0.],decimals=2)

        setattr(amodel, 'teff_0', cal_star_info[8])
        setattr(amodel, 'logg_0', cal_star_info[10])
        setattr(amodel, 'mh_0', cal_star_info[11])

        aw1, af1 = amodel()

        apogee_res1 = starspectrum.flux.value - af1
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / len(apogee_res1)

        chi2_vals += [(np.round_(bosz_chi2,decimals=2),np.round_(0.,decimals=2),np.round_(apogee_chi2,decimals=2))]
        '''
        plt.figure(figsize=(16, 12))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

        plt.text(22090, 0.65,'Star:'+name+'\n$T_{eff}:$' + str(cal_star_info[8]) +'$\pm$'+ str('?') + '\n$log g:$' + str(cal_star_info[10]) +'$\pm$'+ str('?') + \
                 '\n$[M/H]:$' + str(cal_star_info[11]) +'$\pm$'+ str(0.) + '\n'+r'$\alpha$:' + str('?') +'$\pm$'+ str(''),fontsize=12, bbox=props)

        plt.text(22090, 0.4,'BOSZ fit offsets:\n$\Delta T_{eff}:$' + str(bosz_deltas[0]) +'$\pm$'+ str(bosz_sigmas_one[0]) + '\n$\Delta log g:$' + str(bosz_deltas[1]) +'$\pm$'+ str(bosz_sigmas_one[1]) + \
                 '\n$\Delta [M/H]:$' + str(bosz_deltas[2]) +'$\pm$'+ str(bosz_sigmas_one[2]) + '\n'+r'[$\alpha$/Fe]:' + str(bosz_deltas[3]) +'$\pm$'+ str(bosz_sigmas_one[3])
                 + '\n'+'$\chi^{2}$:' + str(bosz_chi2) +'\n$\chi^{2}$ at ref. values'+ str(apogee_chi2), fontsize=12, bbox=props)
        #22035
        #plt.text(22090, 0.1,'PHOENIX fit offsets:\n$\Delta T_{eff}:$' + str(phoenix_deltas[0]) +'$\pm$'+ str(phoenix_sigmas_one[0]) + '\n$\Delta log g:$' + str(phoenix_deltas[1]) +'$\pm$'+ str(phoenix_sigmas_one[1]) + \
        #         '\n$\Delta [M/H]:$' + str(phoenix_deltas[2]) +'$\pm$'+ str(phoenix_sigmas_one[2]) + '\n'+r'$\Delta$$\alpha$:' + str(phoenix_deltas[3]) +'$\pm$'+ str(phoenix_sigmas_one[3]),fontsize=12, bbox=props)

        plt.plot(starspectrum.wavelength.value / (bosz_result.median['vrad_2'] / 3e5 + 1.0), starspectrum.flux.value,
                 color='#000000', label='Data',linewidth=5.0)

        plt.plot(bw1 / (bosz_result.median['vrad_2'] / 3e5 + 1.0), bf1, color='#33AFFF', label='BOSZ Model/Residuals',linewidth=5.0)

        #plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), pf2, color='#FEBE4E', label='PHOENIX Model/Residuals',linewidth=5.0)

        plt.plot(bw1 / (bosz_result.median['vrad_2'] / 3e5 + 1.0), bosz_res1, color='#33AFFF',linewidth=5.0)

        #plt.plot(pw2 / (phoenix_result.median['vrad_4'] / 3e5 + 1.0), phoenix_res2, color='#FEBE4E',linewidth=5.0)

        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
        plt.axhline(y=-0.05, color='k', linestyle='--')

        plt.xlim(21000,22900)
        plt.ylim(-0.2,1.3)

        plt.legend(loc='center left', fontsize=16)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        plt.title(result_title+' fits and residuals for '+name)
        #plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=15)
        plt.savefig('/u/rbentley/localcompute/fitting_plots/'+outputpath+'/'+name+'_'+result_title+'_spectrum.png')
        #plt.show()
        plt.clf()
        plt.cla()
        '''
        ap_values['teff'] += [cal_star_info[8]]
        ap_values['logg'] += [cal_star_info[10]]
        ap_values['mh'] += [cal_star_info[11]]
        #ap_values['alpha'] += [0.0]

    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys())+1, ncols=1,figsize=(7.5, 5.5))
    labelsize = 10

    fig.subplots_adjust(hspace=0.025, wspace=0.0)

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    x_axis_b = bosz_offsets['logg']
    x_axis_p = ap_values['mh']

    x_axis_b = np.array([float(i) for i in x_axis_b])

    for i in range(len(ax)-1):
        combined_offsets = bosz_offsets[bosz_offsets.keys()[i]]
        combined_sigmas = bosz_sigmas[phoenix_offsets.keys()[i]]

        b_stdev = np.std(combined_offsets)
        b_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + b_stdev ** 2) for x in combined_sigmas])
        b_offset_mean = np.average(combined_offsets, weights=b_sigmas_stdev ** -2)

        #ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

        #ppopt, ppcov = curve_fit(linear_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

        #ppopt2, ppcov2 = curve_fit(quad_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[phoenix_offsets.keys()[i]], color='#3349FF',marker='.',label='Reference BOSZ grid',ls='none', markersize=12)

        #bpopt, bpcov = curve_fit(linear_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[bosz_offsets.keys()[i]])

        #bpopt2, bpcov2 = curve_fit(quad_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[bosz_offsets.keys()[i]])

        ax[i].axhline(y=b_offset_mean, color='#3349FF', linestyle='--', label='Reference mean offset')
        ax[i].axhspan(b_offset_mean - b_stdev, b_offset_mean + b_stdev,
                      color='#3349FF', alpha=0.2)


        bpopt, bpcov = curve_fit(linear_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[phoenix_offsets.keys()[i]])

        bpopt_c, bpcov_c = curve_fit(constant_fit, x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], sigma = bosz_sigmas[phoenix_offsets.keys()[i]])

        fit_res_b = np.array(bosz_offsets[bosz_offsets.keys()[i]] - linear_fit(x_axis_b, bpopt[0], bpopt[1]))
        chi2_b_fit = np.sum((fit_res_b)**2 / np.array(bosz_sigmas[phoenix_offsets.keys()[i]])**2)
        chi2_b_fit_red = chi2_b_fit/(len(fit_res_b) - len(bpopt))

        fit_res_b_c = np.array(bosz_offsets[bosz_offsets.keys()[i]] - constant_fit(x_axis_b, bpopt_c[0]))
        chi2_b_fit_c = np.sum((fit_res_b_c)**2 / np.array(bosz_sigmas[phoenix_offsets.keys()[i]])**2)
        chi2_b_fit_c_red = chi2_b_fit_c/(len(fit_res_b_c) - len(bpopt_c))

        f_b = (chi2_b_fit_c - chi2_b_fit)
        f_b = f_b/(len(bpopt) - len(bpopt_c))
        f_b = f_b*(len(fit_res_b)-len(bpopt))/chi2_b_fit

        ax[i].tick_params(axis='y', which='major', labelsize=labelsize)
        ax[i].tick_params(axis='x', which='major', labelsize=0.01)
        #ax[i].set_xlim(-1.0, 0.6)

        #connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[phoenix_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])

        #connect_points(ax[i],x_axis_b,x_axis_n, bosz_offsets[phoenix_offsets.keys()[i]],offsets[offsets.keys()[i]])

        ax[i].axhline(y=0., color='k', linestyle='--')

        if (bosz_offsets.keys()[i] is 'teff'):
            print 'hrtr'
            ax[i].plot(np.linspace(xmin,xmax,100), linear_fit(np.linspace(xmin,xmax,100), bpopt[0], bpopt[1]), color='#3349FF',
                       label='Linear fit to offsets', linestyle=':', markersize=12)

        elif (bosz_offsets.keys()[i] is 'mh'):
            ax[i].plot(np.linspace(xmin,xmax,100), linear_fit(np.linspace(xmin,xmax,100), bpopt[0], bpopt[1]), color='#3349FF',
                       label='Linear fit to offsets', linestyle=':', markersize=12)

        else:
            ax[i].plot([], [], color='#3349FF', label='Linear fit to offsets', linestyle=':', markersize=12)

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()
        if f_b >=100000:
            continue
        else:
            ax[i].text((xmax-xmin)*0.9+xmin, (ymax-ymin)*0.25+ymin, '$\sigma$: ' + str(np.round_(b_stdev,decimals=2))+'\nMean: ' + str(np.round_(b_offset_mean,decimals=2))+'\nF-statistic: '+str(np.round_(f_b,decimals=2))+'\n5% crit val: '+str(np.round_(18.513,decimals=2)),fontsize=8, bbox=props)


    #ax[-1].plot(x_axis_n,[x[0] for x in chi2_vals],color='#5cd85e',marker='d',label=outputpath,ls='none', markersize=10)
    #ax[-1].plot(x_axis_b,[x[1] for x in chi2_vals], color='#3349FF',marker='.',label='Unmasked BOSZ grid',ls='none', markersize=18)
    #ax[-1].plot(x_axis_p,[x[2] for x in chi2_vals],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

    ax[i].axhline(y=b_offset_mean, color='#3349FF', linestyle='--', label='Reference mean offset')



    # ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],color='#FF0000',marker='s',label='Unmasked PHOENIX grid',ls='none', markersize=10)

    # ppopt, ppcov = curve_fit(linear_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

    # ppopt2, ppcov2 = curve_fit(quad_fit, x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], sigma = phoenix_sigmas[phoenix_offsets.keys()[i]])

    # connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[phoenix_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])

    # connect_points(ax[i],x_axis_b,x_axis_n, bosz_offsets[phoenix_offsets.keys()[i]],offsets[offsets.keys()[i]])


    #ax[-1].plot(x_axis_p, [x[0] for x in chi2_vals],color='#3349FF',marker='.',ls='none', markersize=18)
    ax[-1].plot(x_axis_b, [x[1] for x in chi2_vals], color='#3349FF',marker='.',ls='none', markersize=12)
    ax[-1].plot(x_axis_b, [x[2] for x in chi2_vals], color='r',marker='.',label='BOSZ grid at SPEX values',ls='none', markersize=9)


    #ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax)-1].set_xlabel('log g offset', fontsize=labelsize)

    ax[0].set_ylabel('log g offset', fontsize=labelsize)

    #ax[0].set_ylim(-1.4,1.4)

    ax[1].set_ylabel('$T_{eff}$ offset', fontsize=labelsize)
    #ax[2].set_ylim(-400,400)
    ax[2].set_ylabel('[M/H] offset', fontsize=labelsize)
    #ax[3].set_ylim(-0.5,0.5)
    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=labelsize)

    ax[0].legend(fontsize=7,loc='upper left')
    ax[-1].legend(fontsize=7)
    ax[-1].tick_params(axis='y', which='major', labelsize=labelsize)
    ax[-1].tick_params(axis='x', which='major', labelsize=labelsize)
    #ax[-1].set_xlim(-1.0, 0.6)
    #ax[0].set_title('StarKit-APOGEE fit offsets for SPEX stars vs APOGEE [M/H]')
    #plt.show()
    plt.savefig('/u/rbentley/localcompute/fitting_plots/' + outputpath + '/' + outputpath + '_offsets_no_alpha_logg_offset.pdf', bbox_inches='tight')



def plot_nifs_stars(bulge=False, dgalaxy=False):
    plt.figure(figsize=(6, 4))
    nifs_paths = glob.glob('/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/nifs/*.h5')

    nirspec_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'
    gc_stars = ['NE_1_001', 'NE_1_002', 'NE_1_003', 'E7_1_001', 'E7_2_001', 'E7_1_002', 'E7_1_003', 'N2_1_001',
                'E5_1_001', 'N2_1_002', 'N2_1_003']

    mh_offset = 0.14
    mh_scatter = 0.16

    alpha_offset = 0.16
    alpha_scatter = 0.18

    teff = []
    teff_err = []

    logg = []
    logg_err = []

    mh = []
    mh_err = []

    alpha = []
    alpha_err = []

    nn_teff = {}
    nn_teff_err = {}

    nn_logg = {}
    nn_logg_err = {}

    nn_mh = {}
    nn_mh_err = {}

    nn_alpha = {}
    nn_alpha_err = {}

    nirspec_teff = []
    nirspec_teff_err = []

    nirspec_logg = []
    nirspec_logg_err = []

    nirspec_mh = []
    nirspec_mh_err = []

    nirspec_alpha = []
    nirspec_alpha_err = []

    if bulge:
        apogee_bulge = pd.read_csv('/u/rbentley/Downloads/APOGEE_DR16_inner2deg_csv.dat')
        bulge_mh = apogee_bulge['M_H']
        bulge_alpha = apogee_bulge['ALPHA_M']
        bulge_ah = apogee_bulge['ALPHA_M'] - apogee_bulge['M_H']

        plt.errorbar(bulge_mh, bulge_alpha, color='g', marker='.', label='APOGEE (Inner 2 deg)', ls='none',
                     markersize=4, alpha=0.4)

    if dgalaxy:
        sagdeg_vals = list(scipy.genfromtxt('/u/rbentley/sagdeg_abundances_monaco2005.txt', delimiter='\t', skip_header=1, dtype=float))
        sagdeg_vals2 = list(scipy.genfromtxt('/u/rbentley/sagdeg_abundances_bonifacio2004.txt', skip_header=0, dtype=float))

        sagdeg_alpha = np.array([x[3]/2.+x[2]/2. for x in sagdeg_vals])
        sagdeg_mh = np.array([x[1] for x in sagdeg_vals])

        sagdeg_alpha2 = np.array([x[3]/3.+x[5]/2. for x in sagdeg_vals2])
        sagdeg_mh2 = np.array([x[1] for x in sagdeg_vals2])

        Vizier.ROW_LIMIT = 100000
        catalog_list = Vizier.find_catalogs('Hill, 2019')
        catalog = Vizier.get_catalogs(catalog_list.keys())['J/A+A/626/A15/tablec5']

        masked = np.ma.nonzero(catalog['__Mg_Fe_'])
        sculptor_mg = catalog['__Mg_Fe_'][masked]
        sculptor_ca = catalog['__Ca_Fe_'][masked]
        sculptor_mh = catalog['__Fe_H_'][masked]
        sculptor_alpha = np.mean(np.array([sculptor_mg, sculptor_ca]), axis=0)

        fornax = scipy.genfromtxt('/u/rbentley/fornax_abundances_letarte2010.txt', skip_header=1, dtype=None)
        fornax_mh = np.array([x[8] for x in fornax])
        fornax_alpha = np.array([x[6]/2.+x[2]/2. for x in fornax])

        carina = scipy.genfromtxt('/u/rbentley/carina_abundances_norris2017.txt', skip_header=3, dtype=None, delimiter='\t')
        carina_mh = np.array([x[1] for x in carina])
        carina_alpha = np.array([np.mean([x[3],x[5]]) for x in carina])

        plt.errorbar(sagdeg_mh, sagdeg_alpha, xerr=0., yerr=0., color = 'g', marker = 'X', label = 'Sgr dSph Stars', ls = 'none', markersize = 10, alpha=0.4) #(Monaco et al. 2005, Bonifacio et al. 2004)
        plt.errorbar(sagdeg_mh2, sagdeg_alpha2, xerr=0., yerr=0., color = 'g', marker = 'X', ls = 'none', markersize = 10, alpha=0.4) #

        plt.errorbar(sculptor_mh, sculptor_alpha, xerr=0., yerr=0., color = '#59bf43', marker = '.', label = 'Sculptor Dwarf Stars', ls = 'none', markersize = 10, alpha=0.4) #(Hill et al. 2019)

        plt.errorbar(fornax_mh, fornax_alpha, xerr=0., yerr=0., color = '#59bf43', marker = '>', label = 'Fornax Dwarf Stars', ls = 'none', markersize = 10, alpha=0.4) #(Letarte et al. 2010)

        plt.errorbar(carina_mh, carina_alpha, xerr=0., yerr=0., color = '#59bf43', marker = 'v', label = 'Carina Dwarf Stars', ls = 'none', markersize = 10, alpha=0.4) #(Norris et al. 2010)

    for path in nifs_paths:
        result = MultiNestResult.from_hdf5(path)
        sigmas = result.calculate_sigmas(1)

        obs_twice = [ele for ele in gc_stars if (ele in path)]

        for gc in gc_stars:
            if gc in path:
                star = gc

        if obs_twice:
            nn_teff.update({star:result.median['teff_0']})
            nn_teff_err.update({star:(sigmas['teff_0'][1]-sigmas['teff_0'][0])/2.})

            nn_logg.update({star:result.median['logg_0']})
            nn_logg_err.update({star:(sigmas['logg_0'][1]-sigmas['logg_0'][0])/2.})

            nn_mh.update({star:result.median['mh_0']+mh_offset})
            nn_mh_err.update({star:np.sqrt(((sigmas['mh_0'][1]-sigmas['mh_0'][0])/2.)**2 + mh_scatter**2)})

            nn_alpha.update({star:result.median['alpha_0']+alpha_offset})
            nn_alpha_err.update({star:np.sqrt(((sigmas['alpha_0'][1]-sigmas['alpha_0'][0])/2.)**2 + alpha_scatter**2)})

        else:
            teff += [result.median['teff_0']]
            teff_err += [(sigmas['teff_0'][1]-sigmas['teff_0'][0])/2.]

            logg += [result.median['logg_0']]
            logg_err += [(sigmas['logg_0'][1]-sigmas['logg_0'][0])/2.]

            mh += [result.median['mh_0']+mh_offset]
            mh_err += [np.sqrt(((sigmas['mh_0'][1]-sigmas['mh_0'][0])/2.)**2 + mh_scatter**2)]

            alpha += [result.median['alpha_0']+alpha_offset]
            alpha_err += [np.sqrt(((sigmas['alpha_0'][1]-sigmas['alpha_0'][0])/2.)**2 + alpha_scatter**2)]


    for a in nn_teff.keys():
        print a
        result = MultiNestResult.from_hdf5(nirspec_path+a+'_order34-36_bosz_adderr.h5')
        sigmas = result.calculate_sigmas(1)

        nirspec_teff += [result.median['teff_0']]
        nirspec_teff_err += [(sigmas['teff_0'][1]-sigmas['teff_0'][0])/2.]

        nirspec_logg += [result.median['logg_0']]
        nirspec_logg_err += [(sigmas['logg_0'][1]-sigmas['logg_0'][0])/2.]

        nirspec_mh += [result.median['mh_0']+mh_offset]
        nirspec_mh_err += [np.sqrt(((sigmas['mh_0'][1]-sigmas['mh_0'][0])/2.)**2 + mh_scatter**2)]

        nirspec_alpha += [result.median['alpha_0']+alpha_offset]
        nirspec_alpha_err += [np.sqrt(((sigmas['alpha_0'][1]-sigmas['alpha_0'][0])/2.)**2 + alpha_scatter**2)]

        print nn_mh[a], nn_alpha[a], result.median['mh_0'], result.median['alpha_0']

        #connectpoints_plt(nn_mh[a], nn_alpha[a], result.median['mh_0']+mh_offset, result.median['alpha_0']+alpha_offset)
        #plt.errorbar(nn_mh[a], nn_alpha[a], xerr=nn_mh_err[a], yerr=nn_alpha_err[a], fmt='.', c='k', alpha=0.3)
        #plt.scatter(nn_mh[a], nn_alpha[a], c=nn_teff[a], vmin=np.amin(teff), vmax=np.amax(teff), marker='.',
        #            s=300, cmap=plt.cm.get_cmap('plasma'))

    glob_cat = ascii.read('/u/rbentley/glob_info_full.txt')

    mh_flags = glob_cat['f_[Fe/H]']
    glob_mh = glob_cat['[Fe/H]']
    glob_alpha = glob_cat['[a/Fe]']
    glob_names = glob_cat['Name']

    glob_mean_idx = np.where((mh_flags=='b')&(glob_names != 'Pal 6'))

    glob_mh = glob_mh[glob_mean_idx]
    glob_alpha = glob_alpha[glob_mean_idx]
    glob_names = glob_names[glob_mean_idx]

    dg_glob_mean_idx = np.where(((glob_names=='Rup 106') | (glob_names=='Pal 12') | (glob_names=='Ter 7')))

    dg_glob_mh = glob_mh[dg_glob_mean_idx]
    dg_glob_alpha = glob_alpha[dg_glob_mean_idx]
    dg_glob_names = glob_names[dg_glob_mean_idx]

    glob_mh = np.delete(glob_mh, dg_glob_mean_idx)
    glob_alpha = np.delete(glob_alpha, dg_glob_mean_idx)
    glob_names = np.delete(glob_names, dg_glob_mean_idx)

    #plt.errorbar(mh, np.array(alpha), xerr=0, yerr=0,fmt = '.', c='k', alpha=0.3)
    #plt.scatter(mh, np.array(alpha), c=teff, vmin=np.amin(teff), vmax=np.amax(teff), marker = '.', label = 'NIFS BOSZ fits', s = 300, cmap=plt.cm.get_cmap('plasma'))

    plt.errorbar(nirspec_mh, np.array(nirspec_alpha), xerr=np.sqrt(np.array(nirspec_alpha_err)**2 + np.array(nirspec_mh_err)**2), yerr=nirspec_alpha_err,fmt = 'd', c='k', alpha=1., markersize=12, label = 'Galactic center stars')
    #plt.scatter(nirspec_mh, np.array(nirspec_alpha), marker = 'd', label = 'NIRSPEC BOSZ fits', s = 200)

    plt.errorbar(glob_mh,glob_alpha, xerr=0.05, yerr=0.1, color='#FF0000',marker='s',label='Globular Clusters',ls='none', markersize=6, alpha=0.3)
    #plt.errorbar(-0.76,0.26, xerr=0.05, yerr=0.1, color='k',marker='s',label='M71',ls='none', markersize=12, alpha=0.7)


    plt.errorbar(dg_glob_mh,dg_glob_alpha, xerr=0.05, yerr=0.1, color='b',marker='D',label='Globular Clusters associated with dwarf galaxies',ls='none', markersize=6, alpha=0.3)

    plt.legend(fontsize=8)
    plt.xlabel('[M/H]',fontsize=10)
    plt.ylabel(r'[$\alpha$/Fe]',fontsize=10)

    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.tick_params(axis='x', which='major', labelsize=10)

    plt.ylim(-0.5,0.64)

    plt.axhline(y=0., color='k', linestyle='--')
    plt.axvline(x=0., color='k', linestyle='--')

    print nirspec_teff, nirspec_teff_err
    print nirspec_logg, nirspec_logg_err
    print nirspec_mh, nirspec_mh_err
    print nirspec_alpha, nirspec_alpha_err

    if dgalaxy:
        plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/alpha_mh_gc_stars_dgalaxy.pdf', bbox_inches='tight')

    else:
        plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/alpha_mh_gc_stars.pdf', bbox_inches='tight')


def plot_all_gc_spectra():
    gc_stars = ['NE_1_001', 'NE_1_002', 'NE_1_003', 'E7_1_001', 'E7_2_001', 'E7_1_002', 'E7_1_003', 'N2_1_001',
                'E5_1_001', 'N2_1_002', 'N2_1_003', 'S1-23']
    gc_info = scipy.genfromtxt('/u/rbentley/metallicity/NIRSPEC_GC_Targets_info.tsv', delimiter='\t', dtype=None,
                               skip_header=1)

    snr = 30.

    gc_nirspec = None
    star_names = [x[13] for x in gc_info]

    for star in gc_stars:
        for tab_star in star_names:
            if star in tab_star:
                star_info = list(gc_info[star_names.index(tab_star)])

        if gc_nirspec is None:
            gc_nirspec = star_info
        else:
            gc_nirspec = np.vstack([gc_nirspec, star_info])

    gc_nirspec = list(gc_nirspec)

    for i in range(len(gc_nirspec)):
        gc_nirspec[i][13] = gc_stars[i]
    gc_nirspec.sort(key=lambda x: float(x[11]))

    plt.figure(figsize=(7.5, 7.5))

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    for i in range(len(gc_nirspec)):
        star_info = gc_nirspec[i]
        starname = star_info[13]

        save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/'

        fitpath = save_path + starname + '_order34-36_bosz_adderr.h5'

        result = MultiNestResult.from_hdf5(fitpath)

        file1 = glob.glob('/u/rbentley/metallicity/spectra/' + starname + '_order34*.dat')
        file2 = glob.glob('/u/rbentley/metallicity/spectra/' + starname + '_order35*.dat')
        file3 = glob.glob('/u/rbentley/metallicity/spectra/' + starname + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        plt.plot(starspectrum34.wavelength.value / (result.median['vrad_3'] / 3e5 + 1.0),
                 starspectrum34.flux.value + 0.7 * i,
                 color='#000000', label='Data', linewidth=1.0)
        plt.plot(starspectrum35.wavelength.value / (result.median['vrad_4'] / 3e5 + 1.0),
                 starspectrum35.flux.value + 0.7 * i,
                 color='#000000', linewidth=1.0)
        plt.plot(starspectrum36.wavelength.value / (result.median['vrad_5'] / 3e5 + 1.0),
                 starspectrum36.flux.value + 0.7 * i,
                 color='#000000', linewidth=1.0)

        plt.text(21985, 0.8 + 0.7 * i,
                 starname + '\nNIFS [M/H]: ' + gc_nirspec[i][11],
                 fontsize=8, bbox=props) #+ '\nK-mag: ' + gc_nirspec[i][12]

    plt.xlim(21800, 22000)

    plt.xlabel('Wavelength (Angstroms)', fontsize=10)
    plt.ylabel('Normalized Flux', fontsize=10)
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.tick_params(axis='x', which='major', labelsize=10)

    plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/all_gc_spectra.pdf', bbox_inches='tight')


def make_convolved_offset(grids=None):
    snr = 30.

    outputpath = 'plots_for_paper'

    result_title = 'BOSZ'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/' + outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/' + outputpath)

    cal_star_info_all = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))

    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[:-1]]

    textsize = 10

    bosz_vals = {'teff': [], \
                 'logg': [], \
                 'mh': [], \
                 'alpha': []}

    bosz_offsets = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    bosz_sigmas = {'teff': [], \
                   'logg': [], \
                   'mh': [], \
                   'alpha': []}

    phoenix_vals = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    phoenix_offsets = {'teff': [], \
                       'logg': [], \
                       'mh': [], \
                       'alpha': []}

    phoenix_sigmas = {'teff': [], \
                      'logg': [], \
                      'mh': [], \
                      'alpha': []}

    ap_values = {'teff': [], \
                 'logg': [], \
                 'mh': [], \
                 'alpha': []}


    chi2_vals = []
    koa_chi2_vals = []

    if grids is not None:
        phoenix = grids[1]
        bosz = grids[0]

    else:
        phoenix = load_full_grid_phoenix()

        bosz = load_full_grid_bosz()
        # bosz = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w15000_17000_R25000.h5')

    print cal_star_names
    for name in cal_star_names:
        star_ind = cal_star_names.index(name)
        cal_star_info = cal_star_info_all[star_ind]

        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/convolved_4000.0_'+name+'_order34-36_bosz_adderr.h5')

        print bosz_result
        bosz_bounds = bosz_result.calculate_sigmas(1)

        bosz_sigmas['teff'] += [np.sqrt(
            ((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2)]
        bosz_sigmas['logg'] += [np.sqrt(
            ((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[8] ** 2 + 0.1 ** 2)]
        bosz_sigmas['mh'] += [
            np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[6] ** 2 + 0.03 ** 2)]
        bosz_sigmas['alpha'] += [np.sqrt(
            ((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[9] ** 2 + 0.03 ** 2)]

        bosz_sigmas_one = [np.sqrt(
            ((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2), + \
                               np.sqrt(((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                                   8] ** 2 + 0.1 ** 2), + \
                               np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                                   6] ** 2 + 0.03 ** 2), + \
                               np.sqrt(
                                   ((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[
                                       9] ** 2 + 0.03 ** 2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)


        bosz_vals['teff'] += [bosz_result.median['teff_0']]
        bosz_vals['logg'] += [bosz_result.median['logg_0']]
        bosz_vals['mh'] += [bosz_result.median['mh_0']]
        bosz_vals['alpha'] += [bosz_result.median['alpha_0']]

        bosz_offsets['teff'] += [bosz_result.median['teff_0'] - cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0'] - cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0'] - cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0'] - cal_star_info[4]]

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        bmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz, convolve=4000.)

        phoenix_chi2 = 0.0

        amodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1, bw2, bf2, bw3, bf3 = bmodel()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1) + len(bosz_res2) + len(bosz_res3))

        bosz_deltas = np.around(
            [bosz_result.median['teff_0'] - cal_star_info[2], bosz_result.median['logg_0'] - cal_star_info[3],
             bosz_result.median['mh_0'] - cal_star_info[1], bosz_result.median['alpha_0'] - cal_star_info[4]],
            decimals=2)

        setattr(amodel, 'teff_0', cal_star_info[2])
        setattr(amodel, 'logg_0', cal_star_info[3])
        setattr(amodel, 'mh_0', cal_star_info[1])
        setattr(amodel, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = amodel()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))

        chi2_vals += [
            (np.round_(bosz_chi2, decimals=2), np.round_(phoenix_chi2, decimals=2), np.round_(apogee_chi2, decimals=2))]

        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]


    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys()) + 1, ncols=1, figsize=(7.5, 9))

    fig.subplots_adjust(hspace=0.025, wspace=0.0)

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    x_axis_b = ap_values['mh']

    x_axis_b = np.array([float(i) for i in x_axis_b])



    for i in range(len(ax) - 1):
        combined_offsets = np.array(bosz_offsets[bosz_offsets.keys()[i]])
        combined_sigmas = np.array(bosz_sigmas[bosz_offsets.keys()[i]])


        b_stdev = np.std(combined_offsets)
        b_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + b_stdev ** 2) for x in combined_sigmas])
        b_offset_mean = np.average(combined_offsets, weights=b_sigmas_stdev ** -2)

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[bosz_offsets.keys()[i]],
                       color='#3349FF', marker='.', label='StarKit offset', ls='none', markersize=16)


        bpopt, bpcov = curve_fit(linear_fit, x_axis_b, combined_offsets, sigma=combined_sigmas)

        bpopt_c, bpcov_c = curve_fit(constant_fit, x_axis_b, combined_offsets, sigma=combined_sigmas)

        fit_res_b = np.array(combined_offsets - linear_fit(x_axis_b, bpopt[0], bpopt[1]))
        chi2_b_fit = np.sum((fit_res_b) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_red = chi2_b_fit / (len(fit_res_b) - len(bpopt))

        fit_res_b_c = np.array(combined_offsets - constant_fit(x_axis_b, bpopt_c[0]))
        chi2_b_fit_c = np.sum((fit_res_b_c) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_c_red = chi2_b_fit_c / (len(fit_res_b_c) - len(bpopt_c))

        f_b = (chi2_b_fit_c - chi2_b_fit)
        f_b = f_b / (len(bpopt) - len(bpopt_c))
        f_b = f_b * (len(fit_res_b) - len(bpopt)) / chi2_b_fit

        pval = stats.f.sf(f_b, 2, 1)

        ax[i].axhline(y=b_offset_mean, color='#3349FF', linestyle='--', label='Mean offset')
        ax[i].axhspan(b_offset_mean - b_stdev, b_offset_mean + b_stdev,
                      color='#3349FF', alpha=0.2)

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()

        ytext = ymax / 2.

        ax[i].text((xmax-xmin)*0.97+xmin, (ymax-ymin)*0.4+ymin, 'BOSZ $\sigma$:' + str(np.round_(b_stdev, decimals=2)) + '\nBOSZ Mean:' + str(
                np.round_(b_offset_mean, decimals=2)), fontsize=10,
                bbox=props)  # +'\np-value: '+str(np.round_(pval,decimals=2), +'\nF-statistic: '+str(np.round_(f_b,decimals=2))+'\n5% crit val: '+str(np.round_(18.513,decimals=2))



        ax[i].axhline(y=0., color='k', linestyle='--')

        ax[i].tick_params(axis='y', which='major', labelsize=10)
        ax[i].tick_params(axis='x', which='major', labelsize=1)

    ax[-1].tick_params(axis='both', which='major', labelsize=10)

    ax[-1].plot(x_axis_b, [x[0] for x in chi2_vals], color='#3349FF', marker='.', ls='none', markersize=16)
    ax[-1].plot(x_axis_b, [x[2] for x in chi2_vals], color='r', marker='.', label='BOSZ grid at APOGEE values',
                ls='none', markersize=16)
    connect_points(ax[-1], x_axis_b, x_axis_b, [x[0] for x in chi2_vals],[x[2] for x in chi2_vals])



    # ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax) - 1].set_xlabel('APOGEE [M/H]', fontsize=11)

    ax[0].set_ylabel('log g offset', fontsize=11)
    ax[1].set_ylabel(r'[$\alpha$/Fe]' + ' offset', fontsize=11)
    ax[2].set_ylabel('$T_{eff}$ offset (K)', fontsize=11)
    ax[3].set_ylabel('[M/H] offset', fontsize=11)
    # ax[4].set_ylabel(r'[$\alpha$/H]'+' offset', fontsize=16)
    # ax[4].set_ylim(-0.7,0.7)
    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=11)

    ax[0].set_ylim(-1.5, 1.5)
    ax[1].set_ylim(-0.5, 0.5)
    ax[2].set_ylim(-450, 450)
    ax[3].set_ylim(-0.5, 0.5)

    ax[0].legend(fontsize=10, loc='upper left')
    ax[-1].legend(fontsize=10, loc='upper left')
    #ax[0].set_title('StarKit-APOGEE fit offsets with KOA spectra')

    plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/grid_convolved_offsets_bosz.pdf',
                        bbox_inches='tight')


def make_hband_offset(grids=None):
    snr = 30.

    outputpath = 'plots_for_paper'

    result_title = 'BOSZ'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/' + outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/' + outputpath)

    cal_star_info_all = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))

    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[:-1]]

    cal_star_specs = [('NGC6791_J19205+3748282', 'apStar-r8-2M19205338+3748282.fits', 0.44),
                 ('NGC6791_J19213390+3750202', 'apStar-r8-2M19213390+3750202.fits', 0.33), \
                 ('NGC6819_J19413439+4017482', 'apStar-r8-2M19413439+4017482.fits', 0.11),
                 ('M71_J19534827+1848021', 'apStar-r8-2M19534827+1848021.fits', -0.7), \
                 ('M5 J15190+0208', 'apStar-r8-2M15190324+0208032.fits', -1.26),
                 ('TYC 3544', 'apStar-r8-2M18482584+4828027.fits', 0.46),
                 ('NGC6819_J19411+4010517', 'apStar-r8-2M19411705+4010517.fits', 0.05)]

    bosz_vals = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    bosz_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_vals = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_offsets = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    phoenix_sigmas = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    ap_values = {'teff':[],\
               'logg':[],\
               'mh':[],\
               'alpha':[]}

    chi2_vals = []

    if grids is None:
        bosz = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w15000_17000_R25000.h5')
        phoenix = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w15000_17000_R40000_apogee.h5')
    else:
        bosz = grids[0]
        phoenix = grids[1]

    for starname in cal_star_names:
        star_ind = cal_star_names.index(starname)
        cal_star_info = cal_star_info_all[star_ind]


        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/apogee/BOSZ_fits/one_order/bosz_badpix_masked_apogee_one_order_' + starname + '_adderr.h5')
        phoenix_result = MultiNestResult.from_hdf5('/u/rbentley/metallicity/spectra_fits/masked_fit_results/apogee/PHOENIX_fits/one_order/phoenix_badpix_masked_apogee_one_order_' + starname + '_adderr_snr30.h5')

        bosz_bounds = bosz_result.calculate_sigmas(1)

        bosz_vals['teff'] += [bosz_result.median['teff_0']]
        bosz_vals['logg'] += [bosz_result.median['logg_0']]
        bosz_vals['mh'] += [bosz_result.median['mh_0']]
        bosz_vals['alpha'] += [bosz_result.median['alpha_0']]

        phoenix_vals['teff'] += [phoenix_result.median['teff_0']]
        phoenix_vals['logg'] += [phoenix_result.median['logg_0']]
        phoenix_vals['mh'] += [phoenix_result.median['mh_0']]
        phoenix_vals['alpha'] += [phoenix_result.median['alpha_0']]

        bosz_sigmas['teff'] += [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2)]
        bosz_sigmas['logg'] += [np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2)]
        bosz_sigmas['mh'] += [np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2)]
        bosz_sigmas['alpha'] += [np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        bosz_sigmas_one = [np.sqrt(((bosz_bounds['teff_0'][1]-bosz_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((bosz_bounds['logg_0'][1]-bosz_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((bosz_bounds['mh_0'][1]-bosz_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((bosz_bounds['alpha_0'][1]-bosz_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        bosz_sigmas_one = np.around(bosz_sigmas_one, decimals=2)

        print bosz_sigmas_one, cal_star_info

        phoenix_bounds = phoenix_result.calculate_sigmas(1)

        phoenix_sigmas['teff'] += [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2)]
        phoenix_sigmas['logg'] += [np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2)]
        phoenix_sigmas['mh'] += [np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2)]
        phoenix_sigmas['alpha'] += [np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        phoenix_sigmas_one = [np.sqrt(((phoenix_bounds['teff_0'][1]-phoenix_bounds['teff_0'][0])/2)**2 + cal_star_info[7]**2), +\
                           np.sqrt(((phoenix_bounds['logg_0'][1]-phoenix_bounds['logg_0'][0])/2)**2 + cal_star_info[8]**2), +\
                           np.sqrt(((phoenix_bounds['mh_0'][1]-phoenix_bounds['mh_0'][0])/2)**2 + cal_star_info[6]**2),+\
                           np.sqrt(((phoenix_bounds['alpha_0'][1]-phoenix_bounds['alpha_0'][0])/2)**2 + cal_star_info[9]**2)]

        phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)

        bosz_offsets['teff'] += [bosz_result.median['teff_0']-cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0']-cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0']-cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0']-cal_star_info[4]]

        phoenix_offsets['teff'] += [phoenix_result.median['teff_0']-cal_star_info[2]]
        phoenix_offsets['logg'] += [phoenix_result.median['logg_0']-cal_star_info[3]]
        phoenix_offsets['mh'] += [phoenix_result.median['mh_0']-cal_star_info[1]]
        phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0']-cal_star_info[4]]

        for x in cal_star_specs:
            if starname == x[0]:
                spectrafile = x[1]

        hdu = fits.open('/u/rbentley/metallicity/spectra/apogee_spec/'+spectrafile)

        pixmaskhdu = hdu[3].data
        pixmask = np.around(np.log10(pixmaskhdu[1, :]) / np.log10(2))

        uncerthdu = hdu[2].data
        uncert = uncerthdu[1, :]

        fluxhdu = hdu[1].data
        flux_all = fluxhdu[1, :]

        header = fits.getheader('/u/rbentley/metallicity/spectra/apogee_spec/'+spectrafile)
        start_wavelength = header['CRVAL1']  # http://localhost:8888/notebooks/Metallicity%20Analysis.ipynb#
        number_of_bins = header['NWAVE']
        bin_size = header['CDELT1']
        end_wavelength = start_wavelength + (number_of_bins - 1) * bin_size
        wavelength = np.linspace(start_wavelength, end_wavelength, number_of_bins)
        wavelength = np.power(10, wavelength)

        flux = np.delete(flux_all,
                         np.argwhere(
                             (pixmask != float("-inf")) & (pixmask != 9.) & (pixmask != 10.) & (pixmask != 11.)))
        wavelength = np.delete(wavelength, np.argwhere(
            (pixmask != float("-inf")) & (pixmask != 9.) & (pixmask != 10.) & (pixmask != 11.)))
        uncert = np.delete(uncert,
                           np.argwhere(
                               (pixmask != float("-inf")) & (pixmask != 9.) & (pixmask != 10.) & (pixmask != 11.)))

        flux1 = np.delete(flux, np.argwhere((wavelength < 15148.3) | (wavelength > 15810) | (flux == 0.)))
        flux2 = np.delete(flux, np.argwhere((wavelength < 15863.3) | (wavelength > 16150) | (flux == 0.)))
        flux3 = np.delete(flux, np.argwhere((wavelength < 16150) | (wavelength > 16435) | (flux == 0.)))

        wavelength1 = np.delete(wavelength, np.argwhere((wavelength < 15148.3) | (wavelength > 15810) | (flux == 0.)))
        wavelength2 = np.delete(wavelength, np.argwhere((wavelength < 15863.3) | (wavelength > 16150) | (flux == 0.)))
        wavelength3 = np.delete(wavelength, np.argwhere((wavelength < 16150) | (wavelength > 16435) | (flux == 0.)))

        uncert1 = np.delete(uncert, np.argwhere((wavelength < 15148.3) | (wavelength > 15810) | (flux == 0.)))
        uncert2 = np.delete(uncert, np.argwhere((wavelength < 15863.3) | (wavelength > 16150) | (flux == 0.)))
        uncert3 = np.delete(uncert, np.argwhere((wavelength < 16150) | (wavelength > 16435) | (flux == 0.)))

        flux1 = flux1 / np.median(flux1)
        flux2 = flux2 / np.median(flux2)
        flux3 = flux3 / np.median(flux3)

        if snr is None:
            uncert1 = uncert1 / np.median(flux1)
            uncert2 = uncert2 / np.median(flux2)
            uncert3 = uncert3 / np.median(flux3)
        else:
            uncert1 = np.zeros(len(flux1)) + 1.0 / np.float(snr)
            uncert2 = np.zeros(len(flux2)) + 1.0 / np.float(snr)
            uncert3 = np.zeros(len(flux3)) + 1.0 / np.float(snr)

        flux = np.concatenate((flux1, flux2, flux3))
        wavelength = np.concatenate((wavelength1, wavelength2, wavelength3))
        uncert = np.concatenate((uncert1, uncert2, uncert3))

        starspectrum = Spectrum1D.from_array(dispersion=wavelength, flux=flux,
                                             dispersion_unit=u.angstrom,
                                             uncertainty=uncert)

        interp1 = Interpolate(starspectrum)
        convolve1 = InstrumentConvolveGrating.from_grid(bosz, R=21000)
        rot1 = RotationalBroadening.from_grid(bosz, vrot=np.array([10.0]))
        norm1 = Normalize(starspectrum, 4)

        bmodel = bosz | rot1 | DopplerShift(vrad=0) | \
                convolve1 | interp1 | \
                norm1

        amodel = bmodel

        interp1 = Interpolate(starspectrum)
        convolve1 = InstrumentConvolveGrating.from_grid(phoenix, R=21000)
        rot1 = RotationalBroadening.from_grid(phoenix, vrot=np.array([10.0]))
        norm1 = Normalize(starspectrum, 4)

        pmodel = phoenix | rot1 | DopplerShift(vrad=0) | \
                convolve1 | interp1 | \
                norm1

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1 = bmodel()
        bosz_res1 = starspectrum.flux.value - bf1

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum.uncertainty.value) ** 2) / (len(bosz_res1))


        for a in phoenix_result.median.keys():
            setattr(pmodel, a, phoenix_result.median[a])

        pw1, pf1 = pmodel()
        phoenix_res1 = starspectrum.flux.value - pf1

        phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum.uncertainty.value) ** 2) / (len(phoenix_res1))

        phoenix_deltas = np.around([phoenix_result.median['teff_0'] - cal_star_info[2], phoenix_result.median['logg_0'] - cal_star_info[3],
                  phoenix_result.median['mh_0'] - cal_star_info[1], phoenix_result.median['alpha_0'] - cal_star_info[4]],decimals=2)

        bosz_deltas = np.around([bosz_result.median['teff_0'] - cal_star_info[2], bosz_result.median['logg_0'] - cal_star_info[3],
                  bosz_result.median['mh_0'] - cal_star_info[1], bosz_result.median['alpha_0'] - cal_star_info[4]],decimals=2)

        setattr(amodel, 'teff_0', cal_star_info[2])
        setattr(amodel, 'logg_0', cal_star_info[3])
        setattr(amodel, 'mh_0', cal_star_info[1])
        setattr(amodel, 'alpha_0', cal_star_info[4])

        aw1, af1 = amodel()

        apogee_res1 = starspectrum.flux.value - af1

        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum.uncertainty.value) ** 2) / (len(apogee_res1))

        chi2_vals += [(np.round_(bosz_chi2,decimals=2),np.round_(phoenix_chi2,decimals=2),np.round_(apogee_chi2,decimals=2))]

        '''

        plt.figure(figsize=(16, 12))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

        plt.text(16185, 0.7,
                 'APOGEE fitted parameters:\n$T_{eff}:$' + str(cal_star_info[2]) +'$\pm$'+ str(cal_star_info[7]) + '\n$log g:$' + str(cal_star_info[3]) +'$\pm$'+ str(cal_star_info[8]) + \
                 '\n$[M/H]:$' + str(cal_star_info[1]) +'$\pm$'+ str(cal_star_info[6]) + '\n'+r'$\alpha$:' + str(cal_star_info[4]) +'$\pm$'+ str(cal_star_info[9]) + '\n$\chi^2$ at APOGEE values:' + str(np.round_(apogee_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.text(16185, 0.45,
                 'BOSZ fit offsets:\n$\Delta T_{eff}:$' + str(bosz_deltas[0]) +'$\pm$'+ str(bosz_sigmas_one[0]) + '\n$\Delta log g:$' + str(bosz_deltas[1]) +'$\pm$'+ str(bosz_sigmas_one[1]) + \
                 '\n$\Delta [M/H]:$' + str(bosz_deltas[2]) +'$\pm$'+ str(bosz_sigmas_one[2]) + '\n'+r'$\Delta$$\alpha$:' + str(bosz_deltas[3]) +'$\pm$'+ str(bosz_sigmas_one[3]) + '\n$\chi^2$:' + str(np.round_(bosz_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.text(16185, 0.2,
                 'PHOENIX fit offsets:\n$\Delta T_{eff}:$' + str(phoenix_deltas[0]) +'$\pm$'+ str(phoenix_sigmas_one[0]) + '\n$\Delta log g:$' + str(phoenix_deltas[1]) +'$\pm$'+ str(phoenix_sigmas_one[1]) + \
                 '\n$\Delta [M/H]:$' + str(phoenix_deltas[2]) +'$\pm$'+ str(phoenix_sigmas_one[2]) + '\n'+r'$\Delta$$\alpha$:' + str(phoenix_deltas[3]) +'$\pm$'+ str(phoenix_sigmas_one[3]) + '\n$\chi^2$:' + str(np.round_(phoenix_chi2,decimals=2)),
                 fontsize=12, bbox=props)

        plt.plot(starspectrum.wavelength.value / (bosz_result.median['vrad_2'] / 3e5 + 1.0), starspectrum.flux.value,
                 color='#000000', label='Data',linewidth=5.0)


        plt.plot(bw1 / (bosz_result.median['vrad_2'] / 3e5 + 1.0), bf1, color='#33AFFF', label='BOSZ Model/Residuals',linewidth=5.0)

        plt.plot(pw1 / (phoenix_result.median['vrad_2'] / 3e5 + 1.0), pf1, color='#FEBE4E', label='PHOENIX Model/Residuals',linewidth=5.0)

        plt.plot(bw1 / (bosz_result.median['vrad_2'] / 3e5 + 1.0), bosz_res1, color='#33AFFF',linewidth=5.0)

        plt.plot(pw1 / (phoenix_result.median['vrad_2'] / 3e5 + 1.0), phoenix_res1, color='#FEBE4E',linewidth=5.0)

        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%')
        plt.axhline(y=-0.05, color='k', linestyle='--')

        plt.xlim(16080,16200)
        plt.ylim(-0.2,1.3)

        plt.legend(loc='center left', fontsize=16)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        plt.title(result_title+' fits and residuals for '+starname)
        plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, size=15)
        plt.savefig('/u/rbentley/localcompute/fitting_plots/'+outputpath+'/'+starname+'_'+result_title+'_spectrum.pdf')
        plt.clf()
        '''
        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]



    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys()) + 1, ncols=1, figsize=(7.5, 9))

    fig.subplots_adjust(hspace=0.025, wspace=0.0)

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    x_axis_b = ap_values['mh']
    x_axis_p = ap_values['mh']

    x_axis_b = np.array([float(i) for i in x_axis_b])
    x_axis_p = np.array([float(i) for i in x_axis_p])


    for i in range(len(ax) - 1):
        combined_offsets = np.array(bosz_offsets[bosz_offsets.keys()[i]])
        combined_sigmas = np.array(bosz_sigmas[bosz_offsets.keys()[i]])


        b_stdev = np.std(combined_offsets)
        b_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + b_stdev ** 2) for x in combined_sigmas])
        b_offset_mean = np.average(combined_offsets, weights=b_sigmas_stdev ** -2)

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[bosz_offsets.keys()[i]],
                       color='#3349FF', marker='.', label='StarKit offset', ls='none', markersize=16)


        combined_offsets_p = np.array(phoenix_offsets[phoenix_offsets.keys()[i]])
        combined_sigmas_p = np.array(phoenix_sigmas[phoenix_offsets.keys()[i]])


        p_stdev = np.std(combined_offsets_p)
        p_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + p_stdev ** 2) for x in combined_sigmas_p])
        p_offset_mean = np.average(combined_offsets_p, weights=p_sigmas_stdev ** -2)


        ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],
                           color='#FEBE4E', marker='.', ls='none', markersize=16)

        connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[bosz_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])


        bpopt, bpcov = curve_fit(linear_fit, x_axis_b, combined_offsets, sigma=combined_sigmas)

        bpopt_c, bpcov_c = curve_fit(constant_fit, x_axis_b, combined_offsets, sigma=combined_sigmas)

        fit_res_b = np.array(combined_offsets - linear_fit(x_axis_b, bpopt[0], bpopt[1]))
        chi2_b_fit = np.sum((fit_res_b) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_red = chi2_b_fit / (len(fit_res_b) - len(bpopt))

        fit_res_b_c = np.array(combined_offsets - constant_fit(x_axis_b, bpopt_c[0]))
        chi2_b_fit_c = np.sum((fit_res_b_c) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_c_red = chi2_b_fit_c / (len(fit_res_b_c) - len(bpopt_c))

        f_b = (chi2_b_fit_c - chi2_b_fit)
        f_b = f_b / (len(bpopt) - len(bpopt_c))
        f_b = f_b * (len(fit_res_b) - len(bpopt)) / chi2_b_fit

        pval = stats.f.sf(f_b, 2, 1)

        ax[i].axhline(y=b_offset_mean, color='#3349FF', linestyle='--', label='Mean offset')
        ax[i].axhspan(b_offset_mean - b_stdev, b_offset_mean + b_stdev,
                      color='#3349FF', alpha=0.2)

        ax[i].axhline(y=p_offset_mean, color='#FEBE4E', linestyle='--')
        ax[i].axhspan(p_offset_mean - p_stdev, p_offset_mean + p_stdev,
                          color='#FEBE4E', alpha=0.2)

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()

        ytext = ymax / 2.

        ax[i].text((xmax-xmin)*0.97+xmin, (ymax-ymin)*0.4+ymin, 'BOSZ $\sigma$:' + str(np.round_(b_stdev, decimals=2)) + '\nBOSZ Mean:' +
                       str(np.round_(b_offset_mean, decimals=2)) + '\nPHOENIX $\sigma$:' + str(np.round_(p_stdev, decimals=2)) +
                        '\nPHOENIX Mean:' + str(np.round_(p_offset_mean, decimals=2)), fontsize=10,
                       bbox=props)

        ax[i].axhline(y=0., color='k', linestyle='--')

        ax[i].tick_params(axis='y', which='major', labelsize=10)
        ax[i].tick_params(axis='x', which='major', labelsize=1)

    ax[-1].tick_params(axis='both', which='major', labelsize=10)

    ax[-1].plot(x_axis_b, [x[0] for x in chi2_vals], color='#3349FF', marker='.', ls='none', markersize=16)

    ax[-1].plot(x_axis_b, [x[2] for x in chi2_vals], color='r', marker='.', label='BOSZ grid at APOGEE values',
                ls='none', markersize=16)

    connect_points(ax[-1], x_axis_b, x_axis_p, [x[0] for x in chi2_vals],[x[2] for x in chi2_vals])

    ax[-1].plot(x_axis_p, [x[1] for x in chi2_vals], color='#FEBE4E', marker='.', ls='none', markersize=16)

    # ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax) - 1].set_xlabel('APOGEE [M/H]', fontsize=11)

    ax[0].set_ylabel('log g offset', fontsize=11)
    ax[1].set_ylabel(r'[$\alpha$/Fe]' + ' offset', fontsize=11)
    ax[2].set_ylabel('$T_{eff}$ offset (K)', fontsize=11)
    ax[3].set_ylabel('[M/H] offset', fontsize=11)
    # ax[4].set_ylabel(r'[$\alpha$/H]'+' offset', fontsize=16)
    # ax[4].set_ylim(-0.7,0.7)
    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=11)

    #ax[0].set_ylim(-1.5, 1.5)
    #ax[1].set_ylim(-0.5, 0.5)
    #ax[2].set_ylim(-450, 450)
    #ax[3].set_ylim(-0.5, 0.5)

    ax[0].legend(fontsize=10, loc='upper left')
    ax[-1].legend(fontsize=10, loc='upper left')
    #ax[0].set_title('StarKit-APOGEE fit offsets with KOA spectra')

    plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/grid_offsets_hband.pdf',
                        bbox_inches='tight')

def make_gc_spec_plots(grid=None, include_spec_text=False, plot_order=35):

    gc_stars = ['NE_1_001', 'NE_1_002', 'NE_1_003', 'E7_1_001', 'E7_2_001', 'E7_1_002', 'E7_1_003', 'N2_1_001',
                'E5_1_001', 'N2_1_002', 'N2_1_003', 'S1-23']

    snr = 30.

    outputpath = 'plots_for_paper'

    result_title = 'BOSZ'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/' + outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/' + outputpath)


    textsize = 10
    textsize_spec = 7

    if plot_order==34:
        spec_min = 22540
        spec_max = 22700

    elif plot_order==36:
        spec_min = 21240
        spec_max = 21400

    else:
        spec_min = 21940
        spec_max = 22100

    if grid is not None:
        bosz = grid

    else:
        bosz = load_full_grid_bosz()

    for name in gc_stars:
        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + name + '_order34-36_bosz_adderr.h5')

        print bosz_result
        bosz_bounds = bosz_result.calculate_sigmas(1)

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        bmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1, bw2, bf2, bw3, bf3 = bmodel()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1) + len(bosz_res2) + len(bosz_res3))

        plt.figure(figsize=(4.25, 3.5))

        props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

        plt.plot(starspectrum34.wavelength.value / (bosz_result.median['vrad_3'] / 3e5 + 1.0),
                 starspectrum34.flux.value,
                 color='#000000', label='GC Star: '+name, linewidth=1.0)
        plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0),
                 starspectrum35.flux.value,
                 color='#000000', linewidth=1.0)
        plt.plot(starspectrum36.wavelength.value / (bosz_result.median['vrad_5'] / 3e5 + 1.0),
                 starspectrum36.flux.value,
                 color='#000000', linewidth=1.0)

        plt.plot(bw1 / (bosz_result.median['vrad_3'] / 3e5 + 1.0), bf1, color='#33AFFF',
                 linewidth=1.0)
        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bf2, color='#33AFFF', label='BOSZ Model/Residuals',
                 linewidth=1.0)
        plt.plot(bw3 / (bosz_result.median['vrad_5'] / 3e5 + 1.0), bf3, color='#33AFFF',
                 linewidth=1.0)

        plt.plot(bw1 / (bosz_result.median['vrad_3'] / 3e5 + 1.0), bosz_res1, color='#33AFFF', linewidth=1.0)
        plt.plot(bw2 / (bosz_result.median['vrad_4'] / 3e5 + 1.0), bosz_res2, color='#33AFFF', linewidth=1.0)
        plt.plot(bw3 / (bosz_result.median['vrad_5'] / 3e5 + 1.0), bosz_res3, color='#33AFFF', linewidth=1.0)

        plt.xticks(np.arange(spec_min, spec_max, step=30))
        plt.tick_params(axis='y', which='major', labelsize=textsize_spec)
        plt.tick_params(axis='x', which='major', labelsize=textsize_spec)


        plt.axhline(y=0.05, color='k', linestyle='--', label='$\pm$ 5%', linewidth=1.0)
        plt.axhline(y=-0.05, color='k', linestyle='--', linewidth=1.0)


        plt.xlim(spec_min, spec_max)
        plt.ylim(-0.2, 1.3)

        plt.legend(loc='center left', fontsize=textsize_spec)
        plt.xlabel('Wavelength (Angstroms)', fontsize=textsize_spec, labelpad=0)
        plt.ylabel('Normalized Flux', fontsize=textsize_spec, labelpad=0)
        #plt.title(result_title + ' fits and residuals for ' + starname)
        plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=textsize_spec)

        plt.savefig(
                '/u/rbentley/localcompute/fitting_plots/plots_for_paper/specs_corners/' + name + '_' + result_title + '_spectrum_o'+str(plot_order)+'.pdf', bbox_inches='tight')

        plt.clf()


def fit_star_one_order_sens_masked(grid=None, sl_cut=6.0, snr=30.0,l1norm=False, masking_param='mh'):

    name = 'NGC6791_J19205+3748282'

    spec_min = 21940
    spec_max = 22000

    textsize_spec=7

    if grid is not None:
        bosz = grid

    else:
        bosz = load_full_grid_bosz()

    bosz_result = MultiNestResult.from_hdf5(
        '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + name + '_order34-36_bosz_adderr.h5')

    print bosz_result
    bosz_bounds = bosz_result.calculate_sigmas(1)

    file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
    file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
    file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

    waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
    waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
    waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

    starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                     wave_range=waverange34)
    starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                     wave_range=waverange35)
    starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                     wave_range=waverange36)

    starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum34.flux.unit
    starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum35.flux.unit
    starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
        snr)) * starspectrum36.flux.unit

    model = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

    if masking_param is 'mh':
        sl_mh1, sl_mh, sl_mh3 = mtf.s_lambda_three_order(model, 'mh', model.mh_0.value, 0.1)

    elif masking_param is 'teff':
        sl_mh1, sl_mh, sl_mh3 = mtf.s_lambda_three_order(model, 'teff', model.teff_0.value, 200)

    elif masking_param is 'logg':
        sl_mh1, sl_mh, sl_mh3 = mtf.s_lambda_three_order(model, 'logg', model.logg_0.value, 0.1)

    elif masking_param is 'alpha':
        sl_mh1, sl_mh, sl_mh3 = mtf.s_lambda_three_order(model, 'alpha', model.alpha_0.value, 0.1)

    else:
        print 'No mask selected'
        return


    mask_sl_f = []
    mask_sl_w = []
    sl_mask_indices = []

    for i in range(len(sl_mh)):
        if abs(sl_mh[i]) >= float(sl_cut):
            mask_sl_f += [starspectrum35.flux.value[i]]
            mask_sl_w += [starspectrum35.wavelength.value[i]]
            sl_mask_indices += [i]

    masked_data_sl_f = np.delete(starspectrum35.flux.value, sl_mask_indices)
    masked_data_sl_w = np.delete(starspectrum35.wavelength.value, sl_mask_indices)
    masked_data_sl_u = np.delete(starspectrum35.uncertainty.value, sl_mask_indices)

    plt.figure(figsize=(4.25, 3.5))

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0),
                 starspectrum35.flux.value,
                 color='#000000', linewidth=1.0, label='Calibrator Star: '+name)

    plt.scatter(masked_data_sl_w / (bosz_result.median['vrad_4'] / 3e5 + 1.0),
                 masked_data_sl_f, s = 3,
                 color='r', label='Masked datapoints')

    plt.plot(starspectrum35.wavelength.value / (bosz_result.median['vrad_4'] / 3e5 + 1.0),
                 np.absolute(sl_mh)/250.,
                 color='g', linewidth=1.0, label='Sensitivity')

    plt.xticks(np.arange(spec_min, spec_max, step=30))
    plt.tick_params(axis='y', which='major', labelsize=textsize_spec)
    plt.tick_params(axis='x', which='major', labelsize=textsize_spec)

    plt.axhline(y=6./250., color='k', linestyle='--', linewidth=1.0, label='Minimum allowed sensivity')


    plt.xlim(spec_min, spec_max)
    plt.ylim(0.0, 1.7)

    plt.legend(loc='upper left', fontsize=textsize_spec)
    plt.xlabel('Wavelength (Angstroms)', fontsize=textsize_spec, labelpad=0)
    plt.ylabel('Normalized Flux', fontsize=textsize_spec, labelpad=0)
    #plt.title(result_title + ' fits and residuals for ' + starname)
    plotlines.oplotlines(angstrom=True, arcturus=True, alpha=0.25, molecules=False, size=textsize_spec)

    plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/specs_corners/' + name + '_BOSZ_sl_masked_spectrum_o35.pdf', bbox_inches='tight')

    plt.clf()

def sl_masked_offset_plot(grids=None, include_phoenix=False):
    snr = 30.

    outputpath = 'plots_for_paper'

    if not os.path.exists('/u/rbentley/localcompute/fitting_plots/' + outputpath):
        os.mkdir('/u/rbentley/localcompute/fitting_plots/' + outputpath)

    cal_star_info_all = list(
        scipy.genfromtxt('/u/rbentley/metallicity/cal_star_info.dat', delimiter='\t', skip_header=1, dtype=None))

    cal_star_info_all.sort(key=lambda x: x[1])
    cal_star_names = [x[0] for x in cal_star_info_all[:-1]]


    bosz_vals = {'teff': [], \
                 'logg': [], \
                 'mh': [], \
                 'alpha': []}

    bosz_offsets = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    bosz_sigmas = {'teff': [], \
                   'logg': [], \
                   'mh': [], \
                   'alpha': []}

    phoenix_vals = {'teff': [], \
                    'logg': [], \
                    'mh': [], \
                    'alpha': []}

    phoenix_offsets = {'teff': [], \
                       'logg': [], \
                       'mh': [], \
                       'alpha': []}

    phoenix_sigmas = {'teff': [], \
                      'logg': [], \
                      'mh': [], \
                      'alpha': []}

    ap_values = {'teff': [], \
                 'logg': [], \
                 'mh': [], \
                 'alpha': []}


    chi2_vals = []

    if grids is not None:
        phoenix = grids[1]
        bosz = grids[0]

    else:
        phoenix = load_full_grid_phoenix()

        bosz = load_full_grid_bosz()
        # bosz = load_grid('/u/rbentley/metallicity/grids/bosz_t3500_7000_w15000_17000_R25000.h5')

    print cal_star_names
    for name in cal_star_names:
        star_ind = cal_star_names.index(name)
        cal_star_info = cal_star_info_all[star_ind]

        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/sl_masked/mh_masked_sl_cutoff_6.0_' + name + '_order34-36_bosz_adderr.h5')

        print bosz_result
        bosz_bounds = bosz_result.calculate_sigmas(1)

        bosz_sigmas['teff'] += [np.sqrt(
            ((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2)]
        bosz_sigmas['logg'] += [np.sqrt(
            ((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[8] ** 2 + 0.1 ** 2)]
        bosz_sigmas['mh'] += [
            np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[6] ** 2 + 0.03 ** 2)]
        bosz_sigmas['alpha'] += [np.sqrt(
            ((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[9] ** 2 + 0.03 ** 2)]

        bosz_sigmas_one = [np.sqrt(
            ((bosz_bounds['teff_0'][1] - bosz_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[7] ** 2 + 25. ** 2), + \
                               np.sqrt(((bosz_bounds['logg_0'][1] - bosz_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                                   8] ** 2 + 0.1 ** 2), + \
                               np.sqrt(((bosz_bounds['mh_0'][1] - bosz_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                                   6] ** 2 + 0.03 ** 2), + \
                               np.sqrt(
                                   ((bosz_bounds['alpha_0'][1] - bosz_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[
                                       9] ** 2 + 0.03 ** 2)]

        if include_phoenix:
            phoenix_result = MultiNestResult.from_hdf5(
                '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/PHOENIX_fits/sl_masked/mh_masked_sl_cutoff_6.0_' + name + '_order34-36_phoenix_adderr.h5')

            phoenix_bounds = phoenix_result.calculate_sigmas(1)

            phoenix_sigmas['teff'] += [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[
                    7] ** 2 + 91.5 ** 2)]  # cal_star_info[7]
            phoenix_sigmas['logg'] += [np.sqrt(
                ((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 + cal_star_info[
                    8] ** 2 + 0.11 ** 2)]  # cal_star_info[8]
            phoenix_sigmas['mh'] += [np.sqrt(
                ((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 + cal_star_info[
                    6] ** 2 + 0.05 ** 2)]  # cal_star_info[6]
            phoenix_sigmas['alpha'] += [np.sqrt(
                ((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 + cal_star_info[
                    9] ** 2 + 0.05 ** 2)]  # cal_star_info[9]

            phoenix_sigmas_one = [np.sqrt(
                ((phoenix_bounds['teff_0'][1] - phoenix_bounds['teff_0'][0]) / 2) ** 2 + cal_star_info[
                    7] ** 2 + 91.5 ** 2), + \
                                      np.sqrt(((phoenix_bounds['logg_0'][1] - phoenix_bounds['logg_0'][0]) / 2) ** 2 +
                                              cal_star_info[8] ** 2 + 0.1 ** 2), + \
                                      np.sqrt(((phoenix_bounds['mh_0'][1] - phoenix_bounds['mh_0'][0]) / 2) ** 2 +
                                              cal_star_info[6] ** 2 + 0.03 ** 2), + \
                                      np.sqrt(((phoenix_bounds['alpha_0'][1] - phoenix_bounds['alpha_0'][0]) / 2) ** 2 +
                                              cal_star_info[9] ** 2 + 0.03 ** 2)]

            phoenix_sigmas_one = np.around(phoenix_sigmas_one, decimals=2)

            phoenix_vals['teff'] += [phoenix_result.median['teff_0']]
            phoenix_vals['logg'] += [phoenix_result.median['logg_0']]
            phoenix_vals['mh'] += [phoenix_result.median['mh_0']]
            phoenix_vals['alpha'] += [phoenix_result.median['alpha_0']]

            phoenix_offsets['teff'] += [phoenix_result.median['teff_0'] - cal_star_info[2]]
            phoenix_offsets['logg'] += [phoenix_result.median['logg_0'] - cal_star_info[3]]
            phoenix_offsets['mh'] += [phoenix_result.median['mh_0'] - cal_star_info[1]]
            phoenix_offsets['alpha'] += [phoenix_result.median['alpha_0'] - cal_star_info[4]]

        bosz_vals['teff'] += [bosz_result.median['teff_0']]
        bosz_vals['logg'] += [bosz_result.median['logg_0']]
        bosz_vals['mh'] += [bosz_result.median['mh_0']]
        bosz_vals['alpha'] += [bosz_result.median['alpha_0']]

        bosz_offsets['teff'] += [bosz_result.median['teff_0'] - cal_star_info[2]]
        bosz_offsets['logg'] += [bosz_result.median['logg_0'] - cal_star_info[3]]
        bosz_offsets['mh'] += [bosz_result.median['mh_0'] - cal_star_info[1]]
        bosz_offsets['alpha'] += [bosz_result.median['alpha_0'] - cal_star_info[4]]

        file1 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order34*.dat')
        file2 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order35*.dat')
        file3 = glob.glob('/u/ghezgroup/data/metallicity/nirspec/spectra/' + name + '_order36*.dat')

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='micron')
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='micron')
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='micron')

        waverange34 = [np.amin(starspectrum34.wavelength.value[:970]), np.amax(starspectrum34.wavelength.value[:970])]
        waverange35 = [np.amin(starspectrum35.wavelength.value[:970]), np.amax(starspectrum35.wavelength.value[:970])]
        waverange36 = [np.amin(starspectrum36.wavelength.value[:970]), np.amax(starspectrum36.wavelength.value[:970])]

        starspectrum34 = read_fits_file.read_nirspec_dat(file1, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange34)
        starspectrum35 = read_fits_file.read_nirspec_dat(file2, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange35)
        starspectrum36 = read_fits_file.read_nirspec_dat(file3, desired_wavelength_units='Angstrom',
                                                         wave_range=waverange36)

        starspectrum34.uncertainty = (np.zeros(len(starspectrum34.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum34.flux.unit
        starspectrum35.uncertainty = (np.zeros(len(starspectrum35.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum35.flux.unit
        starspectrum36.uncertainty = (np.zeros(len(starspectrum36.flux.value)) + 1.0 / np.float(
            snr)) * starspectrum36.flux.unit

        bmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

        if include_phoenix:
            pmodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, phoenix)

            for a in phoenix_result.median.keys():
                setattr(pmodel, a, phoenix_result.median[a])

            pw1, pf1, pw2, pf2, pw3, pf3 = pmodel()
            phoenix_res1 = starspectrum34.flux.value - pf1
            phoenix_res2 = starspectrum35.flux.value - pf2
            phoenix_res3 = starspectrum36.flux.value - pf3

            phoenix_chi2 = np.sum((phoenix_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 + np.sum((phoenix_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
            phoenix_chi2 = phoenix_chi2 / (len(phoenix_res1) + len(phoenix_res2) + len(phoenix_res3))

            phoenix_deltas = np.around(
                [phoenix_result.median['teff_0'] - cal_star_info[2], phoenix_result.median['logg_0'] - cal_star_info[3],
                 phoenix_result.median['mh_0'] - cal_star_info[1], phoenix_result.median['alpha_0'] - cal_star_info[4]],
                decimals=2)

        else:
            phoenix_chi2 = 0.0

        amodel = make_model_three_order(starspectrum34, starspectrum35, starspectrum36, bosz)

        for a in bosz_result.median.keys():
            setattr(bmodel, a, bosz_result.median[a])

        bw1, bf1, bw2, bf2, bw3, bf3 = bmodel()
        bosz_res1 = starspectrum34.flux.value - bf1
        bosz_res2 = starspectrum35.flux.value - bf2
        bosz_res3 = starspectrum36.flux.value - bf3

        bosz_chi2 = np.sum((bosz_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 + np.sum((bosz_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        bosz_chi2 = bosz_chi2 / (len(bosz_res1) + len(bosz_res2) + len(bosz_res3))

        bosz_deltas = np.around(
            [bosz_result.median['teff_0'] - cal_star_info[2], bosz_result.median['logg_0'] - cal_star_info[3],
             bosz_result.median['mh_0'] - cal_star_info[1], bosz_result.median['alpha_0'] - cal_star_info[4]],
            decimals=2)

        setattr(amodel, 'teff_0', cal_star_info[2])
        setattr(amodel, 'logg_0', cal_star_info[3])
        setattr(amodel, 'mh_0', cal_star_info[1])
        setattr(amodel, 'alpha_0', cal_star_info[4])

        aw1, af1, aw2, af2, aw3, af3 = amodel()

        apogee_res1 = starspectrum34.flux.value - af1
        apogee_res2 = starspectrum35.flux.value - af2
        apogee_res3 = starspectrum36.flux.value - af3
        apogee_chi2 = np.sum((apogee_res1) ** 2 / (starspectrum34.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res2) ** 2 / (starspectrum35.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 + np.sum((apogee_res3) ** 2 / (starspectrum36.uncertainty.value) ** 2)
        apogee_chi2 = apogee_chi2 / (len(apogee_res1) + len(apogee_res2) + len(apogee_res3))

        chi2_vals += [(np.round_(bosz_chi2, decimals=2), np.round_(phoenix_chi2, decimals=2), np.round_(apogee_chi2, decimals=2))]

        ap_values['teff'] += [cal_star_info[2]]
        ap_values['logg'] += [cal_star_info[3]]
        ap_values['mh'] += [cal_star_info[1]]
        ap_values['alpha'] += [cal_star_info[4]]


    fig, ax = plt.subplots(nrows=len(bosz_offsets.keys()) + 1, ncols=1, figsize=(7.5, 9))

    fig.subplots_adjust(hspace=0.025, wspace=0.0)

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.)

    x_axis_b = ap_values['mh']
    x_axis_p = ap_values['mh']

    x_axis_b = np.array([float(i) for i in x_axis_b])
    x_axis_p = np.array([float(i) for i in x_axis_p])


    for i in range(len(ax) - 1):
        combined_offsets = np.array(bosz_offsets[bosz_offsets.keys()[i]])
        combined_sigmas = np.array(bosz_sigmas[bosz_offsets.keys()[i]])

        b_stdev = np.std(combined_offsets)
        b_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + b_stdev ** 2) for x in combined_sigmas])
        b_offset_mean = np.average(combined_offsets, weights=b_sigmas_stdev ** -2)

        ax[i].errorbar(x_axis_b, bosz_offsets[bosz_offsets.keys()[i]], yerr=bosz_sigmas[bosz_offsets.keys()[i]],
                       color='#3349FF', marker='.', label='StarKit offset', ls='none', markersize=16)



        if include_phoenix:
            combined_offsets_p = np.array(phoenix_offsets[phoenix_offsets.keys()[i]])
            combined_sigmas_p = np.array(phoenix_sigmas[phoenix_offsets.keys()[i]])

            p_stdev = np.std(combined_offsets_p)
            p_sigmas_stdev = np.array([np.sqrt(float(x) ** 2 + p_stdev ** 2) for x in combined_sigmas_p])
            p_offset_mean = np.average(combined_offsets_p, weights=p_sigmas_stdev ** -2)

            print phoenix_sigmas[phoenix_offsets.keys()[i]]

            ax[i].errorbar(x_axis_p, phoenix_offsets[phoenix_offsets.keys()[i]], yerr=phoenix_sigmas[phoenix_offsets.keys()[i]],
                           color='#FEBE4E', marker='.', ls='none', markersize=16)

            connect_points(ax[i],x_axis_b,x_axis_p, bosz_offsets[bosz_offsets.keys()[i]],phoenix_offsets[phoenix_offsets.keys()[i]])


        bpopt, bpcov = curve_fit(linear_fit, x_axis_b, combined_offsets, sigma=combined_sigmas)

        bpopt_c, bpcov_c = curve_fit(constant_fit, x_axis_b, combined_offsets, sigma=combined_sigmas)

        fit_res_b = np.array(combined_offsets - linear_fit(x_axis_b, bpopt[0], bpopt[1]))
        chi2_b_fit = np.sum((fit_res_b) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_red = chi2_b_fit / (len(fit_res_b) - len(bpopt))

        fit_res_b_c = np.array(combined_offsets - constant_fit(x_axis_b, bpopt_c[0]))
        chi2_b_fit_c = np.sum((fit_res_b_c) ** 2 / combined_sigmas ** 2)
        chi2_b_fit_c_red = chi2_b_fit_c / (len(fit_res_b_c) - len(bpopt_c))

        f_b = (chi2_b_fit_c - chi2_b_fit)
        f_b = f_b / (len(bpopt) - len(bpopt_c))
        f_b = f_b * (len(fit_res_b) - len(bpopt)) / chi2_b_fit

        pval = stats.f.sf(f_b, 2, 1)

        ax[i].axhline(y=b_offset_mean, color='#3349FF', linestyle='--', label='Mean offset')
        ax[i].axhspan(b_offset_mean - b_stdev, b_offset_mean + b_stdev,
                      color='#3349FF', alpha=0.2)

        if include_phoenix:
            ax[i].axhline(y=p_offset_mean, color='#FEBE4E', linestyle='--')
            ax[i].axhspan(p_offset_mean - p_stdev, p_offset_mean + p_stdev,
                          color='#FEBE4E', alpha=0.2)

        xmin, xmax = ax[i].get_xlim()
        ymin, ymax = ax[i].get_ylim()

        ytext = ymax / 2.

        if include_phoenix:
            ax[i].text((xmax-xmin)*0.97+xmin, (ymax-ymin)*0.4+ymin, 'BOSZ $\sigma$:' + str(np.round_(b_stdev, decimals=2)) + '\nBOSZ Mean:' +
                       str(np.round_(b_offset_mean, decimals=2)) + '\nPHOENIX $\sigma$:' + str(np.round_(p_stdev, decimals=2)) +
                        '\nPHOENIX Mean:' + str(np.round_(p_offset_mean, decimals=2)), fontsize=10,
                       bbox=props)

        else:
            ax[i].text((xmax-xmin)*0.97+xmin, (ymax-ymin)*0.4+ymin, 'BOSZ $\sigma$:' + str(np.round_(b_stdev, decimals=2)) + '\nBOSZ Mean:' + str(
                np.round_(b_offset_mean, decimals=2)), fontsize=10,
                bbox=props)  # +'\np-value: '+str(np.round_(pval,decimals=2), +'\nF-statistic: '+str(np.round_(f_b,decimals=2))+'\n5% crit val: '+str(np.round_(18.513,decimals=2))



        ax[i].axhline(y=0., color='k', linestyle='--')

        ax[i].tick_params(axis='y', which='major', labelsize=10)
        ax[i].tick_params(axis='x', which='major', labelsize=1)

    ax[-1].tick_params(axis='both', which='major', labelsize=10)

    ax[-1].plot(x_axis_b, [x[0] for x in chi2_vals], color='#3349FF', marker='.', ls='none', markersize=16)
    ax[-1].plot(x_axis_b, [x[2] for x in chi2_vals], color='r', marker='.', label='BOSZ grid at APOGEE values',
                ls='none', markersize=16)

    connect_points(ax[-1], x_axis_b, x_axis_p, [x[0] for x in chi2_vals],[x[2] for x in chi2_vals])

    if include_phoenix:
        ax[-1].plot(x_axis_p, [x[1] for x in chi2_vals], color='#FEBE4E', marker='.', ls='none', markersize=16)

    # ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=16)
    ax[len(ax) - 1].set_xlabel('APOGEE [M/H]', fontsize=11)

    ax[0].set_ylabel('log g offset', fontsize=11)
    ax[1].set_ylabel(r'[$\alpha$/Fe]' + ' offset', fontsize=11)
    ax[2].set_ylabel('$T_{eff}$ offset (K)', fontsize=11)
    ax[3].set_ylabel('[M/H] offset', fontsize=11)
    # ax[4].set_ylabel(r'[$\alpha$/H]'+' offset', fontsize=16)
    # ax[4].set_ylim(-0.7,0.7)
    ax[-1].set_ylabel('Reduced $\chi^2$', fontsize=11)

    if include_phoenix:
        ax[0].set_ylim(-3.5, 3.5)
        ax[1].set_ylim(-0.7, 0.7)
        ax[2].set_ylim(-1100, 1100)
        ax[3].set_ylim(-0.7, 0.7)

    else:
        ax[0].set_ylim(-1.5, 1.5)
        ax[1].set_ylim(-0.5, 0.5)
        ax[2].set_ylim(-450, 450)
        ax[3].set_ylim(-0.5, 0.5)

    ax[0].legend(fontsize=10, loc='upper left')
    ax[-1].legend(fontsize=10, loc='upper left')
    #ax[0].set_title('StarKit-APOGEE fit offsets with KOA spectra')
    if include_phoenix:
        plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/grid_offsets_sl_masked_bosz_phoenix.pdf',
            bbox_inches='tight')
    else:
        plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/grid_offsets_sl_masked_bosz.pdf',
                        bbox_inches='tight')

def gc_bulge_mh_comparison():
    plt.figure(figsize=(3.75, 3))

    apogee_bulge = pd.read_csv('/u/rbentley/Downloads/APOGEE_DR16_inner2deg_csv.dat')
    bulge_mh = apogee_bulge['M_H']

    plt.hist(bulge_mh, density=True, label='APOGEE DR16 (Inner 2 deg)', alpha=0.8,bins=15)

    catalog_list = Vizier.find_catalogs('Do, 2015')
    catalog = Vizier.get_catalogs(catalog_list.keys())['J/ApJ/809/143/table1']

    nifs_mh = catalog['__M_H_']

    catalog_list = Vizier.find_catalogs('Feldmeier-Krause+, 2017')
    catalog = Vizier.get_catalogs(catalog_list.keys())['J/MNRAS/464/194/tableb1']

    print catalog.keys()

    kmos_mh = catalog['__M_H_']

    plt.hist(kmos_mh, density=True, label='Galactic center', alpha=0.8,bins=15)

    plt.tick_params(axis='y', which='major', labelsize=7)
    plt.tick_params(axis='x', which='major', labelsize=7)

    sagdeg_vals = list(scipy.genfromtxt('/u/rbentley/sagdeg_abundances_monaco2005.txt', delimiter='\t', skip_header=1, dtype=float))
    sagdeg_vals2 = list(scipy.genfromtxt('/u/rbentley/sagdeg_abundances_bonifacio2004.txt', skip_header=0, dtype=float))

    sagdeg_mh = np.array([x[1] for x in sagdeg_vals])

    sagdeg_mh2 = np.array([x[1] for x in sagdeg_vals2])

    sagdeg_mh = np.concatenate((sagdeg_mh, sagdeg_mh2))

    plt.hist(sagdeg_mh, density=True, label='Sagittarius Dwarf Galaxy', alpha=0.5,bins=15)

    plt.legend(fontsize=7)
    plt.xlabel('[M/H]',fontsize=7)
    plt.ylabel('Normalized Counts',fontsize=7)

    plt.tick_params(axis='y', which='major', labelsize=7)
    plt.tick_params(axis='x', which='major', labelsize=7)

    plt.savefig('/u/rbentley/localcompute/fitting_plots/plots_for_paper/mh_gc_bulge_hist_fk2017.pdf', bbox_inches='tight')