import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter, laplace,\
                          convolve, generic_filter
from skimage import io, data_dir
from skimage import color
from skimage.transform import rescale
from skimage.filters import sobel_h, sobel_v

def ACMO(im):
  # Absolute Central Moment (Shirvaikar2004)
  (m, n) = im.shape
  hist, _ = np.histogram(im, bins=256, range=(0, 1))
  hist = hist / (m * n)
  hist = abs(np.linspace(0, 1, num=256) - np.mean(im[:])) * hist
  fm = sum(hist)
  return fm

def BREN(im):
  # Brenner's (Santos97)
  (m, n) = im.shape
  dh = np.zeros((m, n))
  dv = np.zeros((m, n))
  dv[0:m-2,:] = im[2:,:] - im[0:-2,:]
  dh[:,0:n-2] = im[:,2:] - im[:,0:-2]
  fm = np.maximum(dh, dv)
  fm = fm**2
  fm = fm.mean()
  return fm

def CONT(im):
  # Image contrast (Nanda2001)
  contrast = lambda x : np.sum(np.abs(x[:]-x[4]))
  fm = generic_filter(im, contrast, (3, 3))
  fm = fm.mean()
  return fm

def CURV(im):
  # Image Curvature (Helmli2001)
  m1 = np.array([(-1, 0, 1),(-1, 0, 1),(-1, 0, 1)])
  m2 = np.array([( 1, 0, 1),( 1, 0, 1),( 1, 0, 1)])
  p0 = convolve(im, m1, mode='nearest') / 6
  p1 = convolve(im, m1.T, mode='nearest') / 6
  p2 = 3*convolve(im, m2, mode='nearest') / 10 \
      -convolve(im, m2.T, mode='nearest') / 5
  p3 = -convolve(im, m2, mode='nearest') / 5 \
      +3*convolve(im, m2, mode='nearest') / 10
  fm = np.abs(p0) + np.abs(p1) + np.abs(p2) + np.abs(p3)
  fm = fm.mean()
  return fm

def DCTE(im):
  # DCT energy ratio (Shen2006)
  fm = generic_filter(im, dct_ratio, (8, 8))
  fm = fm.mean()
  return fm

def DCTR(im):
  # DCT reduced energy ratio (Lee2009)
  fm = generic_filter(im, re_ratio, (8, 8))
  fm = fm.mean()
  return fm

def GDER(im, s=3):
  # Gaussian derivative (Geusebroek2000)
  fm = gaussian_filter(im, sigma=s)
  fm = fm.mean()
  return fm

def GLVA(im):
  # Graylevel variance (Krotkov86)
  fm = im.std()
  return fm

def GLLV(im):
  # Graylevel local variance (Pech2000)
  lvar = generic_filter(im, np.std, size=3)**2
  fm = lvar.std()**2
  return fm

def GLVN(im):
  # Normalized GLV (Santos97)
  fm = im.std()**2 / im.mean()
  return fm

def GRAE(im):
  # Energy of gradient (Subbarao92a)
  ix = im
  iy = im
  ix[:,:-1] = np.diff(im, 1, 1)
  iy[:-1,:] = np.diff(im, 1, 0)
  fm = ix**2 + iy**2
  fm = fm.mean()
  return fm

def GRAT(im):
  # Thresholded gradient (Snatos97)
  th = 0
  ix = im
  iy = im
  ix[:,:-1] = np.diff(im, 1, 1)
  iy[:-1,:] = np.diff(im, 1, 0)
  fm = np.maximum(np.abs(ix), np.abs(iy))
  fm[fm<th] = 0
  fm = fm.sum() / np.sum(np.concatenate(fm!=0))
  return fm

def GRAS(im):
  # Squared gradient (Eskicioglu95)
  ix = np.diff(im, 1, 1)
  fm = ix**2
  fm = fm.mean()
  return fm

def HELM(im, s=3):
  # Helmli's mean method (Helmli2001)
  u = generic_filter(im, np.mean, (s, s))
  r1 = np.zeros(im.shape)
  nz = im!=0
  r1[nz] = u[nz] / im[nz]

  index = (u>im)

  fm = np.zeros(im.shape)
  nz = r1!=0
  fm[nz] = 1/r1[nz]

  fm[index] = r1[index]
  fm = fm.mean()
  return fm

def HISE(im):
  # Histogram entropy (Krotkov86)
  _, counts = np.unique(im, return_counts=True)
  fm = entropy(counts)
  return fm

def HISR(im):
  # Histogram range (Firestone91)
  fm = im.max() - im.min()
  return fm

def LAPE(im):
  # Energy of laplacian (Subbarao92a)
  fm = laplace(im)
  fm = (fm**2).mean()
  return fm

def LAPM(im):
  # Modified Laplacian (Nayar89)
  m = np.array([(-1, 2, -1)])
  lx = convolve(im, m, mode='nearest')
  ly = convolve(im, m.T, mode='nearest')
  fm = np.abs(lx) + np.abs(ly)
  fm = fm.mean()
  return fm

def LAPV(im):
  # Variance of laplacian (Pech2000)
  fm = laplace(im)
  fm = fm.std()**2
  return fm

def LAPD(im):
  # Diagonal laplacian (Thelen2009)
  m1 = np.array([(-1, 2, -1)])
  m2 = np.array([(0, 0, -1), (0, 2, 0), (-1, 0, 0)]) / np.sqrt(2)
  m3 = np.array([(-1, 0, 0), (0, 2, 0), (0, 0, -1)]) / np.sqrt(2)
  f1 = convolve(im, m1, mode='nearest')
  f2 = convolve(im, m2, mode='nearest')
  f3 = convolve(im, m3, mode='nearest')
  f4 = convolve(im, m1.T, mode='nearest')
  fm = np.abs(f1) + np.abs(f2) + np.abs(f3) + np.abs(f4)
  fm = fm.mean()
  return fm

def SFIL(im, s=3):
  # Steerable filters (Minhas2009)
  # Angles = [0 45 90 135 180 225 270 315]
  sind = lambda x : np.sin(x * np.pi / 180)
  cosd = lambda x : np.cos(x * np.pi / 180)
  r = np.zeros((im.shape[0], im.shape[1], 8))
  r[:,:,0] = gaussian_filter(im, sigma=s, order=(1,0))
  r[:,:,1] = gaussian_filter(im, sigma=s, order=(0,1))
  r[:,:,2] = cosd(45) *r[:,:,0] + sind(45) *r[:,:,1]
  r[:,:,3] = cosd(135)*r[:,:,0] + sind(135)*r[:,:,1]
  r[:,:,4] = cosd(180)*r[:,:,0] + sind(180)*r[:,:,1]
  r[:,:,5] = cosd(225)*r[:,:,0] + sind(225)*r[:,:,1]
  r[:,:,6] = cosd(270)*r[:,:,0] + sind(270)*r[:,:,1]
  r[:,:,7] = cosd(315)*r[:,:,0] + sind(315)*r[:,:,1]
  fm = np.amax(r,2)
  fm = fm.mean()
  return fm

def SFRQ(im):
  # Spatial frequency (Eskicioglu95)
  ix = im
  iy = im
  ix[:,:-1] = np.diff(im, 1, 1)
  iy[:-1,:] = np.diff(im, 1, 0)
  fm = np.sqrt(ix**2 + iy**2)
  fm = fm.mean()
  return fm

def TENG(im):
  # Tenengrad (Krotkov86)
  gx = sobel_v(im)
  gy = sobel_h(im)
  fm = gx**2 + gy**2
  fm = fm.mean()
  return fm

def TENV(im):
  # Tenengrad variance (Pech2000)
  gx = sobel_v(im)
  gy = sobel_h(im)
  fm = gx**2 + gy**2
  fm = fm.std()**2
  return fm

def VOLA(im):
  # Vollath's correlation (Santos97)
  i1 = im.copy()
  i2 = im.copy()
  i1[:-1,:] = im[1:,:]
  i2[:-2,:] = im[2:,:]
  fm = im * (i1-i2)
  fm = fm.mean()
  return fm

def WAVS(im):
  # Sum of Wavelet coeffs (Yang2003)
  c = pywt.wavedec2(im, 'db6', level=1)
  c_arr, c_slices = pywt.coeffs_to_array(c)
  h = c_arr[c_slices[1]['da']]
  v = c_arr[c_slices[1]['ad']]
  d = c_arr[c_slices[1]['dd']]
  fm = np.abs(h) + np.abs(v) + np.abs(d)
  fm = fm.mean()
  return fm

def WAVV(im):
  # Variance of  Wav...(Yang2003)
  wavelet = 'db6'
  level = 1
  mode = 'symmetric'

  coeffs = pywt.wavedec2(im, wavelet, level=level, mode=mode)
  _, ds = coeffs[0], coeffs[1:] # aa, ds
  (da, ad, dd) = ds[0] # pywt -> matlab: aa=a, da=h, ad=v, dd=d
  coeff = {'da': da}
  h = pywt.idwtn(coeff, wavelet, mode=mode)
  coeff = {'ad': ad}
  v = pywt.idwtn(coeff, wavelet, mode=mode)
  coeff = {'dd': dd}
  d = pywt.idwtn(coeff, wavelet, mode=mode)

  fm = np.std(h)**2 + np.std(v)**2 + np.std(d)**2
  return fm

def WAVR(im):
  wavelet = 'db6'
  level = 1
  mode = 'symmetric'

  coeffs = pywt.wavedec2(im, wavelet, level=level, mode=mode)
  _, ds = coeffs[0], coeffs[1:] # aa, ds
  (da, ad, dd) = ds[0] # pywt -> matlab: aa=a, da=h, ad=v, dd=d
  coeff = {'da': da}
  h = pywt.idwtn(coeff, wavelet, mode=mode)
  coeff = {'ad': ad}
  v = pywt.idwtn(coeff, wavelet, mode=mode)
  coeff = {'dd': dd}
  d = pywt.idwtn(coeff, wavelet, mode=mode)

  level = 3
  coeffs = pywt.wavedec2(im, wavelet, level=level, mode=mode)

  a1 = wrcoef2_a(wavelet, mode, coeffs, 2)
  a2 = wrcoef2_a(wavelet, mode, coeffs, 1)
  a3 = wrcoef2_a(wavelet, mode, coeffs, 0)

  a = a1 + a2 + a3
  wh = h**2 + v**2 + d**2
  wh = wh.mean()
  wl = a.mean()
  fm = wh / wl
  return fm

def dct_ratio(m):
  mt = dct(dct(m.T).T)**2
  fm = (np.sum(mt[:])-mt[0]) / mt[0]
  return fm

def re_ratio(m):
  m = dct(dct(m.T).T)
  fm = (m[1]**2 + m[2]**2 + m[8]**2 + m[9]**2 + m[16]**2) / (m[0]**2)
  return fm

def wrcoef2_a(wavelet, mode, coeffs, level):
  a, ds = coeffs[0], coeffs[1:]
  l = 0
  for d in ds:
    if l >= level:
      d = tuple([np.zeros_like(c) for c in d])

    d = tuple(np.asarray(coeff) if coeff is not None else None
              for coeff in d)
    d_shapes = (coeff.shape for coeff in d if coeff is not None)
    try:
      d_shape = next(d_shapes)
    except StopIteration:
      idxs = slice(None), slice(None)
    else:
      idxs = tuple(slice(None, -1 if a_len == d_len + 1 else None)
                    for a_len, d_len in zip(a.shape, d_shape))

    a = pywt.idwt2((a[idxs], d), wavelet, mode)
    l = l + 1
  return a

def normalize(x):
  return (x - x.min()) / (x.max() - x.min())

def plot_compare_2d(im1, im2, fm1, fm2):
  # Create figure
  _, (ax1, ax2) = plt.subplots(1, 2)

  # Add image
  ax1.imshow(im1, cmap='gray')
  ax2.imshow(im2, cmap='gray')

  # Add titles
  ax1.set_title(fm1)
  ax2.set_title(fm2)

  # Hide axis
  plt.axis('off')

  # Show
  plt.show()

if __name__ == '__main__':

  # Read image
  im1 = io.imread(os.path.join(data_dir, 'camera.png'))
  im2 = gaussian_filter(im1, sigma=5)

  # Normalize
  im1 = normalize(im1)
  im2 = normalize(im2)

  # Focus measurement
  fm1 = ACMO(im1)
  fm2 = ACMO(im2)

  # Plot
  plot_compare_2d(im1, im2, fm1, fm2)
