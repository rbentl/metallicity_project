import numpy as np
import pandas as pd
import pylab as plt
import matplotlib
from astropy import units as u
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood, SpectralL1Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest, MultiNestResult
from starkit import assemble_model, operations
from starkit.fitkit import priors
from starkit.base.operations.spectrograph import (Interpolate, Normalize,
                                                  NormalizeParts,InstrumentConvolveGrating)
from starkit.base.operations.stellar import (RotationalBroadening, DopplerShift)
from starkit.fix_spectrum1d import SKSpectrum1D
from specutils import read_fits_file,plotlines,combine_spectra
import numpy as np
import os,scipy
from specutils import Spectrum1D,rvmeasure
import datetime,glob
import model_tester_updated as mt
from matplotlib.backends.backend_pdf import PdfPages
import fit_nirspec_do2018
import sys

star_name = 'NGC6791_J19205+3748282'

order_range = range(34,37)

spec_path = '/u/ghezgroup/data/metallicity/nirspec/spectra/'

save_path = '/u/rbentley/metallicity/spectra_fits/masked_fit_results/orders34-35-36/'

g = load_grid('/u/rbentley/metallicity/grids/phoenix_t2500_6000_w20000_24000_R40000.h5')


fit_nirspec_do2018.fit_star_multi_order(star_name,g,specdir=spec_path,savedir=save_path,teff_range=[2500,6000],logg_range=[0.,4.5],mh_range=[-2.,1.],alpha_range=[-1.,1.],R_fixed=True)
