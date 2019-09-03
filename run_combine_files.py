from nirspec import combine_files
from astropy.io import fits

combine_files.combine_files('flats_on.txt','flats_on.fits')
combine_files.combine_files('flats_off.txt','flats_off.fits')
combine_files.combine_files('arcs.txt','superarcs.fits')
combine_files.combine_files('etalon.txt','superetalon.fits')

# subtract flats_on from flats_off to make superflat
hdu = fits.open('flats_on.fits')
hdu2 = fits.open('flats_off.fits')

hdu.data = hdu[0].data - hdu2[0].data
hdu.writeto('superflat.fits',clobber=True)
