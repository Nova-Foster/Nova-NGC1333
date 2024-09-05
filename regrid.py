import numpy as np
from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube as sc
from matplotlib import pyplot as plt
from astropy.coordinates import Angle
import pyregion


#hdu_12 = fits.open("C:\\Users\\alexf\Japan internship\\NRO45_13CO_Tmb_0p33kms_21.5arcsec.vrad.fits.fits")[0]
#hdu_13 = fits.open("C:\\Users\\alexf\Japan internship\\NRO45_13CO_Tmb_0p33kms_21.5arcsec.vrad.fits.fits")[0]
hdu_18 = fits.open("C:\\Users\\alexf\Japan internship\\NRO_C18O.vrad&restfreq.fits")[0]


hdu_18.header["BMAJ"] = 14/3600
hdu_18.header["BMIN"] = 14/3600
hdu_18.header["BPA"] = 0
cube_18 = sc.read(hdu_18)

#cube_12 = cube_12.with_spectral_unit(u.km/u.s, velocity_convention='radio')
#cube_13 = cube_13.with_spectral_unit(u.km/u.s, velocity_convention='radio')
cube_18 = cube_18.with_spectral_unit(u.km/u.s)
print(cube_18.header['CTYPE3'])
print(cube_18.header['CUNIT3'])
print(cube_18.header['RESTFRQ'])

# 13 CO data from Submillimter telescope
tp_data_path = "C:\\Users\\alexf\Japan internship\\C18O\\ngc1333TP.C18O.cube.valueK.fits"
hdu_tp = fits.open(tp_data_path)[0]
cube_tp = sc.read(hdu_tp)
cube_tp = cube_tp.with_spectral_unit(u.km/u.s,velocity_convention='radio',rest_value=219560358000*u.Hz)




obs_regiontp = pyregion.open("C:\\Users\\alexf\Japan internship\\TP_13CO_region.REG")
# Creating some subcube based on x and y bounds then applying to both SMT and new obs
xlo = Angle("3h27m50s").to_value(u.deg)
xhi = Angle("3h31m0s").to_value(u.deg)
ylo = Angle("30d50m").to_value(u.deg)
yhi = Angle("31d50m").to_value(u.deg)

subcube_tp = cube_tp
subcube_18 = cube_18

# Creating integrated intensity maps from subcubes created above
submom0tp = subcube_tp.moment0()
submom018 = subcube_18.moment0()


r = obs_regiontp.as_imagecoord(submom018.header)

# Convolving new obs to SMT beam
subcube_18 = subcube_18.convolve_to(subcube_tp.beam)

# Reproject coordinates
subcube_18 = subcube_18.reproject(subcube_tp.header)

# Creating new integrated intensity after convolving (and also SMT again?)
submom0tp = subcube_tp.moment0()
submom018 = subcube_18.moment0()
mapnum=111
vmin1, vmax1 = 1e21, 6e23
vmin2, vmax2 = 0, 0.2
max_dist = 15* u.pixel

plt.rcParams["font.family"] = "Serif"

region_filter = r.get_filter()
region_mask = region_filter.mask(submom018)

print(region_mask)
masked_region = np.where(region_mask==True,submom018,np.nan)
plt.imshow(masked_region.data)
submom018.write('NRO45_18CO_conv&reproject_to_TP.mom0.fits',overwrite=True)