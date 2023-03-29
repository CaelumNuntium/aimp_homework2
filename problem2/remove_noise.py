import argparse
import numpy
from scipy import fft
from astropy.io import fits
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("input", action="store", help="input file name")
args = parser.parse_args()
fits_file = fits.open(args.input)
img = fits_file[0].data
ft_img = fft.fft2(img)
fts_img = fft.fftshift(ft_img)

d1 = (804, 881)
d2 = (834, 911)
fts_img[d1] = numpy.median(fts_img[d1[0] - 1:d1[0] + 1, d1[1] - 1:d1[1] + 1])
fts_img[d2] = numpy.median(fts_img[d2[0] - 1:d2[0] + 1, d2[1] - 1:d2[1] + 1])

ft_img = fft.ifftshift(fts_img)
img = numpy.abs(fft.ifft2(fts_img))
hdu = fits.PrimaryHDU(data=img)
fits.HDUList([hdu]).writeto("result.fits", overwrite=True)
