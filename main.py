import numpy as np
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def readfile(path):
  return np.genfromtxt(path, delimiter=';', skip_header=9, encoding='latin1', dtype = None)

def _plot(data, out_file, bbox_inches = 'tight', metadata = dict()):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(data)
  fig.savefig(out_file, bbox_inches = bbox_inches, metadata = {**metadata, 'CreationDate': None})
  plt.close()
  

DIR='test_03'

files = glob.glob(DIR + '/*.csv')
files = sorted(files)
data = np.empty(shape = [0, 2])

for f in files:
  d = readfile(f)
  data = np.append(data, d, axis = 0)
#  _plot(d, f + '.pdf')

_plot(data[:, 1], DIR + '.pdf')
fft = np.fft.fft(data[:, 1])
_plot(fft, DIR + '_fft.pdf')


