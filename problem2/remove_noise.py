import argparse
import numpy
from scipy import fft
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument("input", action="store", help="input file name")
args = parser.parse_args()
fits_file = fits.open(args.input)
img = fits_file[0].data
ft_img = fft.fft2(img)
fts_img = fft.fftshift(ft_img)

# I don't know what to do

ft_img = fft.ifftshift(fts_img)
img = numpy.abs(fft.ifft2(fts_img))
hdu = fits.PrimaryHDU(data=img)
fits.HDUList([hdu]).writeto("result.fits", overwrite=True)
