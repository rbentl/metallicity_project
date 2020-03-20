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
import numpy as np


def gc_results_table():
    f = open('/u/rbentley/localcompute/fitting_plots/plots_for_paper/gc_fit_table_logg.tex', 'w')

    teff_offset = 140.
    teff_scatter = 157.

    logg_offset = 0.15
    logg_scatter = 0.14

    mh_offset = 0.15
    mh_scatter = 0.14

    alpha_offset = 0.18
    alpha_scatter = 0.16

    gc_stars = ['NE_1_001', 'NE_1_002', 'NE_1_003', 'E7_1_001', 'E7_2_001', 'E7_1_002', 'E7_1_003', 'N2_1_001',
                'E5_1_001', 'N2_1_002', 'N2_1_003', 'S1-23']

    for star in gc_stars:
        bosz_result = MultiNestResult.from_hdf5(
            '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/BOSZ_fits/unmasked/' + star + '_order34-36_bosz_logg1.2_adderr.h5')
        sigmas = bosz_result.calculate_sigmas(1)

        teff = [np.round_(bosz_result.median['teff_0'] + teff_offset),
                np.round_(np.sqrt(((sigmas['teff_0'][1] - sigmas['teff_0'][0]) / 2.) ** 2 + teff_scatter ** 2))]
        logg = [np.round_(bosz_result.median['logg_0'] + logg_offset, decimals=2),
                np.round_(np.sqrt(((sigmas['logg_0'][1] - sigmas['logg_0'][0]) / 2.) ** 2 + logg_scatter ** 2), decimals=2)]
        mh = [np.round_(bosz_result.median['mh_0'] + mh_offset, decimals=2),
              np.round_(np.sqrt(((sigmas['mh_0'][1] - sigmas['mh_0'][0]) / 2.) ** 2 + mh_scatter ** 2), decimals=2)]
        alpha = [np.round_(bosz_result.median['alpha_0'] + alpha_offset, decimals=2),
                 np.round_(np.sqrt(((sigmas['alpha_0'][1] - sigmas['alpha_0'][0]) / 2.) ** 2 + alpha_scatter ** 2), decimals=2)]

        f.write(star + ' & #### & ' + str(teff[0]) + '$\pm$' + str(teff[1]) + ' & ' + str(logg[0]) + '$\pm$' + str(
            logg[1]) +
                ' & ' + str(mh[0]) + '$\pm$' + str(mh[1]) + ' & ' + str(alpha[0]) + '$\pm$' + str(alpha[1]) + ' \\\\\n')

    f.close()

