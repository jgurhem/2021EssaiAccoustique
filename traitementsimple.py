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

DIR=args.dir + '/'

# FileDuration = 240 milliseconds
FileDuration = 0.24
ThresholdMax = 0.01
ThresholdMin = -0.01

files = glob.glob(DIR + '/*.csv')
files = sorted(files, key = sort_func)
NAllFiles = len(files)
files = files[args.start:args.end]
data = np.empty(shape = [0, 2])

if len(files) < 1:
  sys.exit(0)

for f in files:
  print(f)
  d = readfile(f)
  data = np.append(data, d, axis = 0)

Ne = len(data)
FileSize = Ne / len(files)

Te = FileDuration / FileSize
print('FileSize', FileSize)
print('delta t', Te)
print('Points', Ne)

Max = max(data[:, 1])
print('amplitude max', Max)

Min = min(data[:, 1])
print('amplitude min', Min)

Coups = 0
y0 = data[:, 1]
for i in y0:
  if i > ThresholdMax or i < ThresholdMin:
    Coups = Coups + 1

CrossThresholdStart = 0
test = True
while y0[CrossThresholdStart] < ThresholdMax and y0[CrossThresholdStart] > ThresholdMin and CrossThresholdStart < len(y0) - 1:
  CrossThresholdStart += 1

CrossThresholdEnd = len(y0) - 1
while y0[CrossThresholdEnd] < ThresholdMax and y0[CrossThresholdEnd] > ThresholdMin and CrossThresholdEnd > 0:
  CrossThresholdEnd -= 1

CrossMax = 0
while y0[CrossMax] != Max and CrossMax < len(y0) - 1:
  CrossMax += 1

print ('CrossThresholdStart', CrossThresholdStart)
print ('CrossMax', CrossMax)
print ('CrossThresholdEnd', CrossThresholdEnd)

Aire = 0
for i in range(len(y0)-1):
	Aire = (y0[i]**2+y0[i+1]**2)/2*Te
print('Energie', Aire)

timeStart = Te * (CrossThresholdStart + args.start * FileSize)
timeMax = Te * (CrossMax + args.start * FileSize)
timeEnd = Te * (CrossThresholdEnd + args.start * FileSize)
duration = timeEnd - timeStart
dureeMontee = timeMax - timeStart

Freqmoy = Coups / duration
print ('Coups', Coups)
print ('Duree', duration)
print ('Temps debut', timeStart)
print ('Temps Max', timeMax)
print ('Temps fin', timeEnd)
print ('Fr??quence moyenne', Freqmoy)
print ('Duree mont??e', dureeMontee)

_plot(np.arange(args.start * FileSize, args.end * FileSize) * Te, data[:, 1], DIR + '_val.pdf')
np.savetxt(DIR + '_val.csv', data[:, 1], delimiter=",")

frequences = np.arange(args.start * FileSize, args.end * FileSize) * 1.0 / (Te * Ne)
spectre = np.fft.fft(data[:, 1]) / Ne
_plot(frequences, np.abs(spectre), DIR + '_fft.pdf')
np.savetxt(DIR + '_fft.csv', np.abs(spectre), delimiter=",")


y0 = np.abs(y0)
min_ = np.min(y0[y0 > 0])
log_Ne = np.log(np.arange(1, Ne + 1))
y0[y0 == 0] = min_
log_y0 = np.log(np.sort(y0)[::-1])
a, b = _plot_reg(log_Ne, log_y0, DIR + '_rif_nn.pdf')
print('slope', a)
print('intercept', b)

# python traitementsimple.py c:\Users\Anne-Claire\Documents\acoustiquedu21 -s 3170 -e 3200