import numpy as np
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re

def readfile(path):
  return np.genfromtxt(path, delimiter=';', skip_header=9, encoding='latin1', dtype = None)

def _plot(x, y, out_file, bbox_inches = 'tight', metadata = dict(), xlim = None):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(x, y)
  if xlim != None:
    plt.xlim(xlim)
  fig.savefig(out_file, bbox_inches = bbox_inches, metadata = {**metadata, 'CreationDate': None})
  plt.close()
  
def sort_func(x):
  r = []
  for s in re.split(r'([^0-9.]+)', x):
    try:
      r.append(float(s))
    except ValueError:
      r.append(s)
  return r

DIR='test_03'

files = glob.glob(DIR + '/*.csv')
files = sorted(files, key = sort_func)
data = np.empty(shape = [0, 2])

for f in files:
  print(f)
  d = readfile(f)
  data = np.append(data, d, axis = 0)

Ne = len(data)
Te = data[1, 0] - data[0, 0]
print(Te)

_plot(np.arange(Ne) * Te, data[:, 1], DIR + '_val.pdf')
np.savetxt(DIR + '_val.csv', data[:, 1], delimiter=",")

spectre = np.fft.fftshift(np.fft.fft(data[:, 1])) / Ne
frequences = np.arange(Ne) * 1.0 / (Te * Ne)

_plot(frequences, np.abs(spectre), DIR + '_fftshit_abs.pdf')
_plot(frequences, np.real(spectre), DIR + '_fftshit_real.pdf')
_plot(frequences, np.imag(spectre), DIR + '_fftshit_imag.pdf')

spectre = np.fft.fft(data[:, 1]) / Ne
_plot(frequences, np.abs(spectre), DIR + '_fft.pdf')
np.savetxt(DIR + '_fft.csv', np.abs(spectre), delimiter=",")

