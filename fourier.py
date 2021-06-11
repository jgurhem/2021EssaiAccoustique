import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def _plot(x, y, out_file, bbox_inches = 'tight', metadata = dict(), xlim = None):
  fig = plt.figure()
  ax = fig.gca()
  ax.plot(x, y)
  if xlim != None:
    plt.xlim(xlim)
  fig.savefig(out_file, bbox_inches = bbox_inches, metadata = {**metadata, 'CreationDate': None})
  plt.close()

def f1(t):
  return np.cos(2 * np.pi * t / 50) + np.sin(2 * np.pi * t / 40)

def f2(t):
  return np.cos(2 * np.pi * t / 50 + np.pi / 3) + np.sin(2 * np.pi * t / 40 + np.pi / 5)

dt = 1
n = 1000
testname = 'fourier1'

T = np.arange(0, n, dt)
V = f1(T)

_plot(T, V, testname + '.pdf')

spectre = np.fft.fftshift(np.fft.fft(V)) / n
frequences = T * 1.0 / (dt * n)

_plot(frequences, np.abs(spectre), testname + '_fftshit_abs.pdf', xlim = [0.45, 0.55])
_plot(frequences, np.real(spectre), testname + '_fftshit_real.pdf', xlim = [0.45, 0.55])
_plot(frequences, np.imag(spectre), testname + '_fftshit_imag.pdf', xlim = [0.45, 0.55])

spectre = np.fft.fft(V) / n
_plot(frequences, np.abs(spectre), testname + '_fft.pdf', xlim = [0, 0.1])

