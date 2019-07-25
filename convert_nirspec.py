from specutils import read_fits_file,plotlines,write_spectrum
from specutils import Spectrum1D,rvmeasure

def mkfits_spectrum(name,order,outfile):
    # Read in NIRSPEC .dat files for a star and order and combine them
    wave = {34:[2.2406, 2.271],
            35:[2.178,2.20882],
            36:[2.11695,2.14566],
            37:[2.0750,2.0875]}
    if order not in wave.keys():
        wave_range = None
    else:
        wave_range = wave[order]
    filename = glob.glob(directory+name+'_order'+str(order)+'*.dat')
    starspectrum = read_fits_file.read_nirspec_dat(filename,wave_range=wave_range,desired_wavelength_units='Angstrom')
    write_spectrum.write_txt(starspectrum.wavelength.value,starspectrum.flux.value,
                             outfile)
    
