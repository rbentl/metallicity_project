import sl_response_plot_multiorder as slp
from specutils import read_fits_file,plotlines
import pylab as plt

mod = slp.load_full_grid()

spec_path = '/u/rbentley/metallicity/spectra/'

ngc6791_282 = 'NGC6791_J19205+3748282'

ngc6819_517 = 'NGC6819_J19411+4010517'

m5_208 = 'M5 J15190+0208'

ngc6791_202 = 'NGC6791_J19213390+3750202'

ngc6819_482 = 'NGC6819_J19413439+4017482'

m71_021 = 'M71_J19534827+1848021'

tyc_3544 = 'TYC 3544'

sl_vals_ngc6791_282, specs_ngc6791_282 = slp.sl_response_plot_four(ngc6791_282, mod, specdir=spec_path)


star_data = specs_ngc6791_282['1.0']

plt.plot(star_data[0][0],star_data[0][1], 'g')
plt.plot(star_data[1][0],star_data[1][1], 'g')
plt.plot(star_data[2][0],star_data[2][1], 'g')
plt.plot(star_data[3][0],star_data[3][1], 'g')

plt.plot(star_data[4][0],star_data[4][1], 'b')
plt.plot(star_data[5][0],star_data[5][1], 'b')
plt.plot(star_data[6][0],star_data[6][1], 'b')
plt.plot(star_data[7][0],star_data[7][1], 'b')

plt.plot(star_data[8][0],star_data[8][1], 'r.')

plotlines.oplotlines(angstrom=True,arcturus=True,molecules=False,alpha=0.5,size=6,highlight=['O','Ne','Mg','Si','S','Ar','Ca','Ti','Cr','Fe','Ni'])

plt.show()
