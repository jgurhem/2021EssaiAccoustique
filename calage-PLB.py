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
from datetime import datetime
from scipy.integrate import simps
from numpy import trapz
import csv

str2date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S %fm')

def readfile(path, mint, maxt):
  # use csv.DictReader() for faster reading
  # https://stackoverflow.com/questions/16503560/read-specific-columns-from-a-csv-file-with-csv-module
  data = np.genfromtxt(path, delimiter=';', skip_header=9, encoding='latin1', dtype = None, usecols=(1))
  v = np.max(data)
  t = np.genfromtxt(path, delimiter=';', skip_header=1, max_rows = 1, encoding='latin1', dtype = None, usecols=(0), converters = {0: str2date})
  coupsint = 0
  for i in range(len(data)):
    if data[i] > maxt or data[i] < mint:
      coupsint = coupsint + 1
  return v, t, data, coupsint

def _plot(x, y, out_file, bbox_inches = 'tight', metadata = dict(), xlim = None, xlabel = None, ylabel = None):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(x, y)
  if xlim != None:
    plt.xlim(xlim)
  if xlabel != None:
    plt.xlabel(xlabel)
  if ylabel != None:
    plt.ylabel(ylabel)
  fig.savefig(out_file, bbox_inches = bbox_inches, metadata = {**metadata, 'CreationDate': None})
  plt.close()

def _plot_reg(x, y, out_file, bbox_inches = 'tight', metadata = dict(), xlim = None, xlabel = None, ylabel = None):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(x, y)
  slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
  ax.plot(x, slope * x + intercept)
  if xlim != None:
    plt.xlim(xlim)
  if xlabel != None:
    plt.xlabel(xlabel)
  if ylabel != None:
    plt.ylabel(ylabel)
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
parser.add_argument('-tmin', type=float, help='threshold min', dest='tmin', required=True)
parser.add_argument('-tmax', type=float, help='threshold max', dest='tmax', required=True)
parser.add_argument('-te', type=float, help='Te', dest='te', required=True)
args = parser.parse_args()

DIR=args.dir + '/'

# FileDuration = 240 milliseconds
ThresholdMax = args.tmin
ThresholdMin = args.tmax

files = glob.glob(DIR + '/*.csv')
files = sorted(files, key = sort_func)
NAllFiles = len(files)
files = files[args.start:args.end]
y0 = []
all_values = []
dates = []
coups = []

Te = args.te

if len(files) < 1:
  sys.exit(0)

for f in files:
  print(f)
  d = readfile(f, ThresholdMin, ThresholdMax)
  y0.append(d[0])
  dates.append(d[1])
  all_values.append(d[2])
  coups.append(d[3])

print(dates)
relative_dates = []
for i in dates:
	delta = i - dates[0]
	relative_dates.append(delta.total_seconds())

all_time = relative_dates[-1]
print("all_time :", all_time/Te)
print("relative_dates :", relative_dates)
print("y0 :", y0)

print(all_values)

valeurs_calees_x = []
valeurs_calees_y = []

for i in range(len(relative_dates)):
	incr = int(relative_dates[i]/Te)
	for j in range(len(all_values[i])):
		valeurs_calees_y.append(all_values[i][j])
		valeurs_calees_x.append((j + incr)*Te)

vals = dict()
for i in range(len(valeurs_calees_y)):
	vals[valeurs_calees_x[i]] = valeurs_calees_y[i]

valeurs_calees_x = vals.keys()
valeurs_calees_y = vals.values()

valeurs_calees_y = [i for _,i in sorted(zip(valeurs_calees_x, valeurs_calees_y))]
valeurs_calees_x = sorted(valeurs_calees_x)

_plot(valeurs_calees_x, valeurs_calees_y, DIR + '_newcalage.pdf', xlabel = "Temps (s)", ylabel = "Amplitude en V")
np.savetxt(DIR + '_newcalage.csv', np.column_stack((valeurs_calees_x, valeurs_calees_y)), delimiter=";")

nombredevaleur = len(valeurs_calees_y)
print('nombre de valeurs', nombredevaleur)


Ne = len(valeurs_calees_y)
frequences = np.array(valeurs_calees_x) * 1.0 / (Te * Ne)
spectre = np.fft.fft(valeurs_calees_y) / Ne
_plot(frequences, np.abs(spectre), DIR + '_fft.pdf', xlabel = "Frequence en MHz", ylabel = "Amplitude en V")
np.savetxt(DIR + '_fft.csv', np.abs(spectre), delimiter=",")


print('delta t', Te)
print('Points', Ne)

Max = max(valeurs_calees_y)
print('amplitude max', Max)

Min = min(valeurs_calees_y)
print('amplitude min', Min)


	
energies = []
for i in range(len(valeurs_calees_y) - 1):
  area = (valeurs_calees_y[i]**2 + valeurs_calees_y[i+1]**2) * (valeurs_calees_x[i+1] - valeurs_calees_x[i]) / 2
  energies.append(area)


_plot(valeurs_calees_x[:-1], energies, DIR + '_calage_energies.pdf', xlabel = "Temps en s", ylabel = "Energie en V^2*s")
np.savetxt(DIR + '_calage_energies.csv', np.column_stack((valeurs_calees_x[:-1], valeurs_calees_y[:-1], energies)), delimiter=";")

Coups = 0
coups = []
for i in valeurs_calees_y:
  if i > ThresholdMax or i < ThresholdMin:
    Coups = Coups + 1
  coups.append(Coups)
	
_plot(valeurs_calees_x, coups, DIR + '_calage_coups.pdf', xlabel = "Temps en s", ylabel = "Nombre de coups")
np.savetxt(DIR + '_calage_coups.csv', np.column_stack((valeurs_calees_x, valeurs_calees_y, coups)), delimiter=";")

CrossThresholdStart = 0
test = True
while valeurs_calees_y[CrossThresholdStart] < ThresholdMax and valeurs_calees_y[CrossThresholdStart] > ThresholdMin and CrossThresholdStart < len(valeurs_calees_y) - 1:
  CrossThresholdStart += 1

CrossThresholdEnd = len(valeurs_calees_y) - 1
while valeurs_calees_y[CrossThresholdEnd] < ThresholdMax and valeurs_calees_y[CrossThresholdEnd] > ThresholdMin and CrossThresholdEnd > 0:
  CrossThresholdEnd -= 1

CrossMax = 0
while valeurs_calees_y[CrossMax] != Max and CrossMax < len(valeurs_calees_y) - 1:
  CrossMax += 1

print ('CrossThresholdStart', CrossThresholdStart)
print ('CrossMax', CrossMax)
print ('CrossThresholdEnd', CrossThresholdEnd)



y0 = np.abs(valeurs_calees_y)
min_ = np.min(y0[y0 > 0])
log_Ne = np.log(np.arange(1, len(valeurs_calees_y) + 1))
y0[y0 == 0] = min_
log_y0 = np.log(np.sort(y0)[::-1])
a, b = _plot_reg(log_Ne, log_y0, DIR + '_rif_nn.pdf')
print('slope', a)
print('intercept', b)

