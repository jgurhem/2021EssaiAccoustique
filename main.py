import numpy as np
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def readfile(path):
  return np.genfromtxt(path, delimiter=';', skip_header=9, encoding='latin1', dtype = None)

def _plot(x, y, out_file, bbox_inches = 'tight', metadata = dict()):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(x, y)
  fig.savefig(out_file, bbox_inches = bbox_inches, metadata = {**metadata, 'CreationDate': None})
  plt.close()
  

DIR='test_03'

files = glob.glob(DIR + '/*.csv')
files = sorted(files)
data = np.empty(shape = [0, 2])

for f in files:
  d = readfile(f)
  data = np.append(data, d, axis = 0)

Ne = len(data)
Te = data[1, 0] - data[0, 0]
print(Te)

_plot(np.arange(Ne) * Te, data[:, 1], DIR + '.pdf')

spectre = np.fft.fftshift(np.fft.fft(data[:, 1])) / Ne
frequences = np.arange(Ne) * 1.0 / (Te * Ne)

_plot(frequences, np.abs(spectre), DIR + '_fftshit_abs.pdf')
_plot(frequences, np.real(spectre), DIR + '_fftshit_real.pdf')
_plot(frequences, np.imag(spectre), DIR + '_fftshit_imag.pdf')

spectre = np.fft.fft(data[:, 1]) / Ne
_plot(frequences, np.abs(spectre), DIR + '_fft.pdf')

