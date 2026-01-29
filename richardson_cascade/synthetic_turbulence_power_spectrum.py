"""
Synthetic turbulence power spectrum figure (frequency and wavenumber versions)

This script generates a stylized spectrum with three regimes:
1) energy input (energy-containing) range (low f or low k),
2) inertial range with Kolmogorov scaling ~ f^{-5/3} or k^{-5/3},
3) dissipation range with a steep exponential roll-off.

It produces two figures:
- PSD vs frequency f (Hz): PSD(f)
- energy spectrum vs wavenumber k (1/length): E(k)

Notes on what to label on the y-axis
- If the x-axis is temporal frequency f (Hz), the natural quantity is a power spectral density PSD(f).
  In many turbulence contexts, one relates PSD(f) to a wavenumber spectrum via Taylor's frozen turbulence hypothesis.
- If the x-axis is spatial wavenumber k, the standard object is the kinetic energy spectrum E(k).

This script is intentionally schematic. It does not depend on measured data.

author: Fabrizio Musacchio
date: Feb 2021 / Jan 2026
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% FUNCTIONS
def spectrum_von_karman_like(
    x,
    x0,
    A=1.0,
    a=4.0,
    b=2.0,
    c=17.0 / 6.0,
    x_d=None,
    beta=1.4,
):
    """
    Von Kármán like spectrum with inertial slope -5/3 and optional dissipation roll off.

    Form:
        S(x) = A * (x/x0)^a / (1 + (x/x0)^b)^c * exp(-(x/xd)^beta)

    Asymptotics:
    - For x << x0:
        S(x) ~ A * (x/x0)^a
      so the spectrum rises toward the energy containing peak.

    - For x >> x0 (before dissipation):
        S(x) ~ A * x^{a - b c}
      With a=4, b=2, c=17/6:
        a - b c = 4 - 2*(17/6) = 4 - 17/3 = -5/3

    Parameters
    ----------
    x : array
        Positive frequency f or positive wavenumber k.
    x0 : float
        Injection scale. Roughly where the energy containing range transitions
        into the inertial range.
    A : float
        Overall amplitude.
    a, b, c : float
        Shape parameters. Default values enforce a rise at small x and -5/3 at large x.
    x_d : float or None
        Dissipation cutoff scale. If None, no dissipation cutoff is applied.
    beta : float
        Dissipation steepness parameter.

    Returns
    -------
    y : array
        Stylized spectrum values.
    """
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-300)

    r = x / float(x0)
    y = A * (r**a) / (1.0 + r**b) ** c

    if x_d is not None:
        y = y * np.exp(- (x / float(x_d)) ** beta)

    return y


def plot_regimes(
    ax,
    x,
    y,
    x_input_end,
    x_inertial_end,
    x_diss_end,
    xlabel,
    ylabel,
    title=None,
    slope_ref=None,
):
    """
    Plot spectrum and shade regimes.

    Regimes:
    - input range:      [x.min(), x_input_end]
    - inertial range:   [x_input_end, x_inertial_end]
    - dissipation:      [x_inertial_end, x_diss_end]

    slope_ref (optional):
    - dict with keys: {"slope": -5/3, "x0": anchor x, "y0": anchor y, "label": "..."}
    """
    ax.loglog(x, y, lw=3, color="black")

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    x_input_end = float(np.clip(x_input_end, xmin, xmax))
    x_inertial_end = float(np.clip(x_inertial_end, xmin, xmax))
    x_diss_end = float(np.clip(x_diss_end, xmin, xmax))

    ax.axvspan(xmin, x_input_end, alpha=0.08)
    ax.axvspan(x_input_end, x_inertial_end, alpha=0.05)
    ax.axvspan(x_inertial_end, x_diss_end, alpha=0.08)

    finite_pos = np.isfinite(y) & (y > 0)
    y_mid = np.sqrt(np.nanmax(y[finite_pos]) * np.nanmin(y[finite_pos]))

    ax.text(np.sqrt(xmin * x_input_end), y_mid, "Energy input\nrange", ha="center", va="top")
    ax.text(np.sqrt(x_input_end * x_inertial_end), y_mid, "Inertial range", ha="center", va="top")
    ax.text(np.sqrt(x_inertial_end * x_diss_end), y_mid, "Dissipation\nrange", ha="center", va="top")

    if slope_ref is not None:
        m = float(slope_ref["slope"])
        x0 = float(slope_ref["x0"])
        y0 = float(slope_ref["y0"])
        label = str(slope_ref.get("label", f"{m:.2f}"))

        x_ref = np.array([x_input_end, x_inertial_end], dtype=float)
        y_ref = y0 * (x_ref / x0) ** m
        ax.loglog(x_ref, y_ref, lw=2, ls="--")

        xm = np.sqrt(x_ref[0] * x_ref[1])
        ym = y0 * (xm / x0) ** m
        ax.text(xm, ym, label, ha="left", va="bottom")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(True, which="both", alpha=0.25)


# %% MAIN SCRIPT

# frequency spectrum (PSD vs f):
f = np.logspace(1, 5, 900)   # 10 Hz ... 100 kHz, schematic
f0 = 80.0                    # injection scale in frequency space
f_d = 1.5e4                  # dissipation cutoff

psd = spectrum_von_karman_like(
    f,
    x0=f0,
    A=3e4,
    a=4.0,
    b=2.0,
    c=17.0 / 6.0,
    x_d=f_d,
    beta=1.6,
)

# Regime boundaries (visual choices)
f_input_end = 1.5e2
f_inertial_end = 7.0e3
f_diss_end = f.max()

# Anchor the reference slope in the inertial range
f_ref_anchor = 1.0e3
psd_ref_anchor = float(np.interp(f_ref_anchor, f, psd))

fig1, ax1 = plt.subplots(figsize=(6, 5))
plot_regimes(ax1, f, psd,
    x_input_end=f_input_end,
    x_inertial_end=f_inertial_end,
    x_diss_end=f_diss_end,
    xlabel="Frequency f (Hz)",
    ylabel="Power spectral density PSD(f)",
    title="Schematic turbulence spectrum in frequency space",
    slope_ref={
        "slope": -5.0 / 3.0,
        "x0": f_ref_anchor,
        "y0": psd_ref_anchor,
        "label": "Kolmogorov slope  -5/3",
    })
plt.savefig("synthetic_turbulence_power_spectrum_frequency.png", dpi=200)
plt.close(fig1)



# wavenumber spectrum (E(k) vs k):
k = np.logspace(0, 4, 900)   # 1 ... 1e4, schematic
k0 = 30.0                    # injection scale in k space
k_d = 2.5e3                  # dissipation cutoff in k space

E_k = spectrum_von_karman_like(
    k,
    x0=k0,
    A=2e2,
    a=4.0,
    b=2.0,
    c=17.0 / 6.0,
    x_d=k_d,
    beta=1.6,
)

k_input_end = 60.0
k_inertial_end = 1.2e3
k_diss_end = k.max()

k_ref_anchor = 2.0e2
Ek_ref_anchor = float(np.interp(k_ref_anchor, k, E_k))

fig2, ax2 = plt.subplots(figsize=(6, 5))
plot_regimes(
    ax2,
    k,
    E_k,
    x_input_end=k_input_end,
    x_inertial_end=k_inertial_end,
    x_diss_end=k_diss_end,
    xlabel="Wavenumber k (1/length)",
    ylabel="Energy spectrum E(k)",
    title="Schematic turbulence spectrum in wavenumber space",
    slope_ref={
        "slope": -5.0 / 3.0,
        "x0": k_ref_anchor,
        "y0": Ek_ref_anchor,
        "label": "Kolmogorov slope  -5/3",
    },
)
plt.savefig("synthetic_turbulence_spectrum_wavenumber.png", dpi=200)
plt.close(fig2)
    
# %% END
