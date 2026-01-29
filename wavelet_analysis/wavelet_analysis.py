""" 
Just a simple demonstration of wavelet analysis (CWT) using PyWavelets.

Generates a synthetic intermittent turbulence-like signal with a known
power-law spectrum, performs CWT using a complex Morlet wavelet, and
visualizes the results including the scalogram and global wavelet spectrum.

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
# %% FUNCTIONS
def colored_noise(alpha: float, n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate 1D colored noise with power spectral density ~ 1 / f^alpha.
    alpha = 0: white noise
    alpha = 1: pink noise
    alpha = 2: brown noise

    Implementation: scale Fourier amplitudes, assign random phases, inverse FFT.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Amplitude ~ 1 / f^(alpha/2) so that power ~ 1 / f^alpha
    amp = np.ones_like(freqs)
    amp[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))

    phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.shape)
    spectrum = amp * (np.cos(phases) + 1j * np.sin(phases))

    x = np.fft.irfft(spectrum, n=n)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x


def intermittent_bursts(t: np.ndarray, rng: np.random.Generator, n_bursts: int = 8) -> np.ndarray:
    """
    Add intermittent, localized packets of band-limited oscillations.
    This mimics bursts where energy temporarily concentrates at specific scales.
    """
    y = np.zeros_like(t)
    T = t[-1] - t[0]

    for _ in range(n_bursts):
        t0 = rng.uniform(t[0] + 0.05 * T, t[0] + 0.95 * T)
        width = rng.uniform(0.03 * T, 0.10 * T)
        f0 = rng.uniform(1.0, 20.0)  # central frequency in Hz
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.8, 2.0)

        envelope = np.exp(-0.5 * ((t - t0) / width) ** 2)
        carrier = np.sin(2.0 * np.pi * f0 * (t - t0) + phase)
        y += amp * envelope * carrier

    y = (y - y.mean()) / (y.std() + 1e-12)
    return y


def periodogram(x: np.ndarray, fs: float):
    """
    Simple periodogram: |FFT|^2, one-sided.
    For real data, Welch's method is typically preferred.
    """
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)

    # Normalization is sufficient for relative comparisons and slope estimation
    Pxx = (np.abs(X) ** 2) / n
    return f, Pxx


def fit_loglog_slope(f: np.ndarray, y: np.ndarray, fmin: float, fmax: float):
    """
    Linear regression in log10 space:
    log10(y) = a + b * log10(f).

    Returns the slope b.
    """
    mask = (f >= fmin) & (f <= fmax) & (y > 0) & np.isfinite(y)
    xf = np.log10(f[mask])
    yy = np.log10(y[mask])

    if len(xf) < 5:
        return np.nan, (np.nan, np.nan)

    b, a = np.polyfit(xf, yy, 1)
    return b, (a, b)
# %% MAIN SCRIPT

# -----------------------------
# 1) synthetic turbulence-like signal
# -----------------------------
rng = np.random.default_rng(4)

fs = 200.0          # sampling frequency in Hz
T = 20.0            # total duration in seconds
n = int(fs * T)
t = np.arange(n) / fs

# Background colored noise with spectral exponent alpha
alpha = 5.0 / 3.0   # commonly associated with inertial-range turbulence
x_bg = colored_noise(alpha=alpha, n=n, fs=fs, rng=rng)

# Add intermittent bursts
x_b = intermittent_bursts(t, rng=rng, n_bursts=10)

# Combine components and add weak white noise
x = 0.9 * x_bg + 0.9 * x_b + 0.1 * rng.standard_normal(n)
x = (x - x.mean()) / (x.std() + 1e-12)


# -----------------------------
# 2) Fourier spectrum
# -----------------------------
f_fft, Pxx = periodogram(x, fs=fs)

# Fit slope over a selected frequency band
slope_fft, _ = fit_loglog_slope(f_fft[1:], Pxx[1:], fmin=2.0, fmax=40.0)


# -----------------------------
# 3) CWT wavelet analysis (improved visualization)
# -----------------------------
# Use a complex Morlet wavelet for a cleaner magnitude representation.
# PyWavelets uses the naming scheme "cmorB-C", where:
# B controls bandwidth, C controls center frequency of the wavelet (not the signal).
wavelet = "cmor1.5-1.0"

# Define frequency range for the scalogram
fmin, fmax = 0.5, 60.0  # Hz
freqs = np.linspace(fmax, fmin, 300)  # higher resolution than before

# Convert target frequencies to scales:
# For CWT in PyWavelets: f = scale2frequency(wavelet, scale) / dt
dt = 1.0 / fs
scales = pywt.frequency2scale(wavelet, freqs * dt)

coeffs, freqs_out = pywt.cwt(x, scales, wavelet, sampling_period=dt)
power = np.abs(coeffs) ** 2

# global wavelet spectrum (time-averaged power per frequency):
global_power = power.mean(axis=1)

# fit slope over the same frequency band for comparability:
slope_wav, _ = fit_loglog_slope(freqs_out, global_power, fmin=2.0, fmax=40.0)

# compute time-resolved energy in a narrow band:
band_lo, band_hi = 0.5, 0.6  # Hz, adjust as needed
band_mask = (freqs_out >= band_lo) & (freqs_out <= band_hi)
band_energy_t = power[band_mask, :].mean(axis=0)

# use log-power (dB) and robust color limits for clearer scalograms:
power_db = 10.0 * np.log10(power + 1e-12)

# robust clipping to improve contrast:
vmin = np.percentile(power_db, 5.0)
vmax = np.percentile(power_db, 99.5)


# -----------------------------
# 3.1) cone of influence (COI)
# -----------------------------
# For the Morlet wavelet, the e-folding time is sqrt(2) * scale
coi_time = np.sqrt(2.0) * scales * dt  # in seconds

t_start = t[0]
t_end = t[-1]

# Left and right COI boundaries
coi_left = t_start + coi_time
coi_right = t_end - coi_time

# %% PLOTS

fig = plt.figure(figsize=(9, 11))
# use GridSpec for flexible layout:
gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    height_ratios=[1.0, 1.0, 1.5],
    hspace=0.45,
    wspace=0.30)

t_min = t[0]
t_max = 20#t[-1]

# top row: signal (full width):
ax_signal = fig.add_subplot(gs[0, :])
ax_signal.plot(t, x, linewidth=1.0)
ax_signal.set_xlabel("time [s]")
ax_signal.set_ylabel("x(t) [a.u.]")
ax_signal.set_title("Synthetic intermittent turbulence-like signal")
ax_signal.set_xlim(t_min, t_max)

# middle row: Fourier spectrum (left):
ax_fft = fig.add_subplot(gs[1, 0])
ax_fft.loglog(f_fft[1:], Pxx[1:], linewidth=1.0)
ax_fft.set_xlabel("frequency f [Hz]")
ax_fft.set_ylabel("power Pxx [a.u.]")
ax_fft.set_title(f"Fourier spectrum\nfitted slope ~ {slope_fft:.2f} (2–40 Hz)")

# middle row: global wavelet spectrum (right):
ax_wav = fig.add_subplot(gs[1, 1])
ax_wav.loglog(freqs_out, global_power, linewidth=1.0)
ax_wav.set_xlabel("frequency [Hz]")
ax_wav.set_ylabel("global wavelet power [a.u.]")
ax_wav.set_title(f"Global wavelet spectrum\nfitted slope ~ {slope_wav:.2f} (2–40 Hz)")

# bottom row: wavelet scalogram (full width):
ax_sca = fig.add_subplot(gs[2, :])
extent = [t[0], t[-1], freqs_out[-1], freqs_out[0]]
im = ax_sca.imshow(
    power_db,
    aspect="auto",
    extent=extent,
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax)
ax_sca.set_xlim(t_min, t_max)
ax_sca.set_yscale("log")
ax_sca.set_xlabel("time [s]")
ax_sca.set_ylabel("frequency [Hz]")
ax_sca.set_title("Wavelet scalogram (CWT power, dB)")
# overlay cone of influence:
ax_sca.plot(coi_left, freqs_out, "w--", linewidth=1.5, label="cone of influence")
ax_sca.plot(coi_right, freqs_out, "w--", linewidth=1.5)
ax_sca.legend(loc="lower left", frameon=False)
# colorbar aligned to scalogram:
# cbar = fig.colorbar(im, ax=ax_sca, pad=0.02)
# cbar.set_label("power [dB]")
# create a new axes below the scalogram for a horizontal colorbar:
cax = fig.add_axes([
    ax_sca.get_position().x0,        # left aligned with scalogram
    ax_sca.get_position().y0 - 0.045, # slightly below scalogram
    ax_sca.get_position().width,     # same width as scalogram
    0.02                              # height of colorbar
])
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("power [dB]")

# finalize and save:
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_PATH, "wavelet_analysis_overview.png"), dpi=300)
plt.close(fig)


# plot time-resolved band energy as a separate figure:
plt.figure(figsize=(9, 3.0))
plt.plot(t, band_energy_t, linewidth=1.0)
plt.xlabel("time [s]")
plt.ylabel("band energy [a.u.]")
plt.title(f"Time-resolved wavelet energy ({band_lo:.2f} to {band_hi:.2f} Hz)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "wavelet_analysis_band_energy.png"), dpi=300)
plt.close()
# %% END