import numpy as np
import pandas as pd
import pylab as plt
import os
from specutils import plotlines

spectra_dir = '../spectra'

def combine():
    # combine the two nods for each of the stars that have two nods

    stars = ['NE_1_002']
    order = [34]
    for i in xrange(len(stars)):

        for j in xrange(len(order)):
            order_n = 'order'+str(order[j])            
            file1 = os.path.join(spectra_dir,stars[i]+'_'+order_n+'_nod1.dat')
            file2 = os.path.join(spectra_dir,stars[i]+'_'+order_n+'_nod2.dat')

            tab1 = pd.read_csv(file1,delim_whitespace=True,skiprows=3,
                               names = ['pixel','wave','flux','nod1','nod2','diff'])
            tab2 = pd.read_csv(file2,delim_whitespace=True,skiprows=3,
                               names = ['pixel','wave','flux','nod1','nod2','diff'])
        
            plt.clf()
            plt.plot(tab1['wave'],tab1['flux'])
            plt.plot(tab2['wave'],tab2['flux'])
            plt.plot(tab1['wave'],(tab2['flux']+tab1['flux'])/2.0+0.2)
            plt.ylim(0.7,1.5)
