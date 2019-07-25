import numpy as np
import pandas as pd
import pylab as plt
import os
from specutils import plotlines

spectra_dir = '../spectra'

def plot_test():
    # plot some of the spectra to test

    file1 = os.path.join(spectra_dir,'NE_1_002_order34_nod1.dat')
    file2 = os.path.join(spectra_dir,'NE_1_002_order34_nod2.dat')    
    file3 = os.path.join(spectra_dir,'NE_1_002_order35_nod1.dat')
    file4 = os.path.join(spectra_dir,'NE_1_002_order35_nod2.dat')
    
    tab1 = pd.read_csv(file1,delim_whitespace=True,skiprows=3,
                       names = ['pixel','wave','flux','nod1','nod2','diff'])
    tab2 = pd.read_csv(file2,delim_whitespace=True,skiprows=3,
                       names = ['pixel','wave','flux','nod1','nod2','diff'])
    tab3 = pd.read_csv(file3,delim_whitespace=True,skiprows=3,
                       names = ['pixel','wave','flux','nod1','nod2','diff'])
    tab4 = pd.read_csv(file4,delim_whitespace=True,skiprows=3,
                       names = ['pixel','wave','flux','nod1','nod2','diff'])
    plt.clf()
    plt.plot(tab1['wave'],tab1['flux'])
    plt.plot(tab2['wave'],tab2['flux'])
    plt.plot(tab3['wave'],tab3['flux'])
    plt.plot(tab4['wave'],tab4['flux'])
    plt.ylim(0,1.5)

def plot_orders(star='NE_1_002',orders=[37,36,35,34],ylim=[0,1.5],noclear=False,vel=0):
    # plot all the orders for a star
    if not noclear:
        plt.clf()
    for i in xrange(len(orders)):
        file1 = os.path.join(spectra_dir,star+'_order'+str(orders[i])+'_nod1.dat')
        tab1 = pd.read_csv(file1,delim_whitespace=True,skiprows=3,
                           names = ['pixel','wave','flux','nod1','nod2','diff'])

        plt.subplot(len(orders),1,i+1)
        plt.plot(tab1['wave'],tab1['flux'],label=star+' order '+str(orders[i]))
        plt.ylim(ylim[0],ylim[1])
        plotlines.oplotlines(vel=vel)
        plt.legend(loc=3)

