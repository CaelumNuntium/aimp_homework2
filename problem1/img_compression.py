import argparse
import warnings
import cv2
import numpy
from scipy import fftpack
from astropy.io import fits
from matplotlib import pyplot as plt

start = 5
step = 5
end = 100 + step

parser = argparse.ArgumentParser()
parser.add_argument("input", action="store", help="input file name")
parser.add_argument("--fits", action="store_true", help="use key --fits if input is FITS-file")
args = parser.parse_args()
if args.fits:
    img_fits = fits.open(args.input)
    img = numpy.array([_.data for _ in img_fits])
    img1 = numpy.ndarray(shape=(img.shape[1], img.shape[2], img.shape[0]), dtype=img.dtype)
    for i in range(img.shape[0]):
        img1[:, :, i] = img[i, :, :]
    img = img1
else:
    img = cv2.imread(args.input)
fourier_img = [fftpack.fft2(img[:, :, i]) for i in range(img.shape[2])]
qualities = []
sigmas = []
for p in range(start, end, step):
    compressed_img = []
    ql = []
    for i in range(len(fourier_img)):
        f_img = fourier_img[i]
        q = numpy.percentile(numpy.abs(f_img), 100 - p)
        c_img = numpy.where(numpy.abs(f_img) > q, f_img, 0)
        compressed_img.append(numpy.abs(fftpack.ifft2(c_img)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ql.append(numpy.sum(numpy.where(img[:, :, i] > 0, (img[:, :, i] - compressed_img[i]) ** 2 / img[:, :, i], 0)))
    qualities.append(numpy.mean(ql))
    sigmas.append(numpy.std(ql))
    if args.fits:
        prim_hdu = fits.PrimaryHDU(data=compressed_img[0])
        hdu = [fits.ImageHDU(data=compressed_img[_]) for _ in range(1, len(compressed_img))]
        hdu_list = [prim_hdu] + hdu
        fits.HDUList(hdu_list).writeto(f"res{p}.fits", overwrite=True)
    else:
        c_res = numpy.array(compressed_img)
        c_res1 = numpy.ndarray(shape=(c_res.shape[1], c_res.shape[2], c_res.shape[0]), dtype=c_res.dtype)
        for i in range(c_res.shape[0]):
            c_res1[:, :, i] = c_res[i, :, :]
        c_res = c_res1
        cv2.imwrite(f"res{p}.bmp", c_res)
plt.xlabel("Proportion of saved coefficients, %")
plt.ylabel("Quality")
plt.errorbar(range(start, end, step), qualities, yerr=sigmas, fmt="o")
fig = plt.gcf()
fig.set_size_inches(12, 5.5)
fig.savefig("result.png", dpi=250)
plt.show()
