""" 
Just a simple demonstration of wavelet functions using PyWavelets.

author: Fabrizio Musacchio
date: Mar 2021 / Jan 2026
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import pywt

RESULTS_PATH = "wavelet_analysis_figures"
os.makedirs(RESULTS_PATH, exist_ok=True)

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% WAVELET PLOTTING

# time axis for wavelet visualization:
t = np.linspace(-6, 6, 2000)

# continuous real wavelets:
real_wavelets = [
    "mexh",     # Mexican hat
    "gaus1",    # Gaussian 1st derivative
    "gaus4",    # Gaussian 4th derivative
    "gaus8"     # Gaussian 8th derivative
]

# continuous complex wavelets:
complex_wavelets = [
    "cmor1.5-1.0",  # Complex Morlet
    "cmor1.0-1.0",  
    "shan1.5-1.0",
    "fbsp2-1.5-1.0"
]

nrows = max(len(real_wavelets), len(complex_wavelets))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=2,
    figsize=(7, 8),
    sharex=True)

# ------------------------------------------------------------
# Real continuous wavelets (left column)
# ------------------------------------------------------------
for i, wname in enumerate(real_wavelets):
    wavelet = pywt.ContinuousWavelet(wname)
    psi, x = wavelet.wavefun(length=len(t))
    axes[i, 0].plot(x, psi, linewidth=1.2)
    axes[i, 0].set_title(wname)
    axes[i, 0].set_ylabel("amplitude")

# ------------------------------------------------------------
# Complex continuous wavelets (right column)
# ------------------------------------------------------------
for i, wname in enumerate(complex_wavelets):
    wavelet = pywt.ContinuousWavelet(wname)
    psi, x = wavelet.wavefun(length=len(t))
    axes[i, 1].plot(x, np.real(psi), label="real", linewidth=1.2)
    axes[i, 1].plot(x, np.imag(psi), "--", label="imag", linewidth=1.2)
    axes[i, 1].set_title(wname)
    axes[i, 1].legend(frameon=False, fontsize=9)

# remove unused axes:
for j in range(len(real_wavelets), nrows):
    axes[j, 0].axis("off")
for j in range(len(complex_wavelets), nrows):
    axes[j, 1].axis("off")

axes[-1, 0].set_xlabel("time")
axes[-1, 1].set_xlabel("time")

plt.suptitle("Common continuous wavelet functions", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "wavelet_functions.png"), dpi=300)
plt.close(fig)

# plot haar wavelet separately due to its discontinuity:
wavelet = pywt.Wavelet("haar")
phi, psi, x = wavelet.wavefun(level=5)
plt.figure(figsize=(3.5, 3))
#plt.plot(x, psi, linewidth=1.5)
plt.step(x, psi, where="post", linewidth=2.0)
plt.xlabel("time")
plt.ylabel("amplitude")
plt.title("Haar wavelet (discrete)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "wavelet_function_haar.png"), dpi=300)
plt.close()

# %% END