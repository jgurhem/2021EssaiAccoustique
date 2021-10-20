import numpy as np
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import re
import argparse
import sys
from scipy import signal
from scipy import stats

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

def _plot_reg(x, y, out_file, bbox_inches = 'tight', metadata = dict(), xlim = None):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(x, y)
  slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
  ax.plot(x, slope * x + intercept)
  if xlim != None:
    plt.xlim(xlim)
  fig.savefig(out_file, bbox_inches = bbox_inches, metadata = {**metadata, 'CreationDate': None})
  plt.close()
  return slope, intercept

def sort_func(x):
  r = []
  for s in re.split(r'([^0-9.]+)', x):
    try:
      r.append(float(s))
    except ValueError:
      r.append(s)
  return r

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='path to the directory containing the csv files')
parser.add_argument('-s', type=int, help='first file to consider', dest='start', default=0)
parser.add_argument('-e', type=int, help='last file to consider', dest='end', default=None)
args = parser.parse_args()

DIR=args.dir

files = glob.glob(DIR + '/*.csv')
files = sorted(files, key = sort_func)
files = files[args.start:args.end]
data = np.empty(shape = [0, 2])

if len(files) < 1:
  sys.exit(0)

for f in files:
  print(f)
  d = readfile(f)
  data = np.append(data, d, axis = 0)

Ne = len(data)
Te = data[1, 0] - data[0, 0]
print('delta t', Te)

Max = max(data[:, 1])
print('amplitude max', Max)

Min = min(data[:, 1])
print('amplitude min', Min)

_plot(np.arange(Ne) * Te, data[:, 1], DIR + '_val.pdf')
np.savetxt(DIR + '_val.csv', data[:, 1], delimiter=",")

frequences = np.arange(Ne) * 1.0 / (Te * Ne)
spectre = np.fft.fft(data[:, 1]) / Ne
_plot(frequences, np.abs(spectre), DIR + '_fft.pdf')
np.savetxt(DIR + '_fft.csv', np.abs(spectre), delimiter=",")

# https://www.f-legrand.fr/scidoc/docimg/sciphys/caneurosmart/pysignal/pysignal.html
# 3.c reduction du bruit
P=20
b1 = signal.firwin(numtaps= 2 * P + 1, cutoff = [0.1], window = 'hann', nyq = 0.5)
a1 = [1.0]
zi = signal.lfiltic(b1, a1, x=[0], y=[0])
[y0, zf] = signal.lfilter(b1, a1, data[:, 1], zi=zi)
_plot(np.arange(Ne) * Te, y0, DIR + '_rif.pdf')
np.savetxt(DIR + '_rif.csv', y0, delimiter=",")

y0 = np.abs(y0)
min_ = np.min(y0[y0 > 0])
log_Ne = np.log(np.arange(1, Ne + 1))
y0[y0 == 0] = min_
log_y0 = np.log(np.sort(y0)[::-1])
a, b = _plot_reg(log_Ne, log_y0, DIR + '_rif_nn.pdf')
print('slope', a)
print('intercept', b)
