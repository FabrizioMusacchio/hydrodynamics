"""
1D hydrostatic atmosphere (dry, ideal gas) with piecewise linear temperature profile.

This script computes standard-style vertical profiles:
  T(h)   temperature [K]
  p(h)   pressure [Pa]
  rho(h) density [kg/m^3]
for altitude h in [0, 86] km, using hydrostatic balance and the ideal gas law.

Physics
  Hydrostatic equilibrium:
    dp/dh = -rho g

  Ideal gas law (dry air):
    p = rho R T

  Combine both:
    dp/dh = -(p g) / (R T(h))

For each atmospheric layer with constant lapse rate L = dT/dh:
  T(h) = T_b + L (h - h_b)

  If L != 0:
    p(h) = p_b * (T(h)/T_b)^(-g/(R L))

  If L == 0 (isothermal):
    p(h) = p_b * exp(-g (h - h_b) / (R T_b))

Then:
  rho(h) = p(h) / (R T(h))

Units
  h in meters internally. Plot uses km.

References
  The layer boundaries and lapse rates follow the 1976 U.S. Standard Atmosphere
  up to 86 km, in a commonly used simplified piecewise form.

Fab note
  This is not a dynamical model. It is a static 1D reference profile.
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% CONSTANTS AND BASE CONDITIONS
# physical constants (standard):
g0 = 9.80665            # m/s^2
R  = 287.05287          # J/(kg K), specific gas constant for dry air

# sea level base conditions:
T0 = 288.15             # K
p0 = 101325.0           # Pa

RESULTS_PATH = "hydrostatic_atmosphere_1D_figures"
os.makedirs(RESULTS_PATH, exist_ok=True)
# %% FUNCTION(S)
def standard_atmosphere_1d(h_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute T(h), p(h), rho(h) for altitudes h_m (meters).

    Parameters
    ----------
    h_m : array_like
        Altitudes in meters. Must be >= 0.

    Returns
    -------
    T : np.ndarray
        Temperature in K.
    p : np.ndarray
        Pressure in Pa.
    rho : np.ndarray
        Density in kg/m^3.
    """
    h_m = np.asarray(h_m, dtype=float)
    if np.any(h_m < 0):
        raise ValueError("Altitudes must be >= 0 m.")

    # Layer definition: base altitude hb [m], top altitude ht [m], lapse rate L [K/m]
    # L = dT/dh (positive means temperature increases with altitude).
    layers = [
        (0e3,   11e3,  -6.5e-3),   # Troposphere
        (11e3,  20e3,   0.0),      # Tropopause region (isothermal)
        (20e3,  32e3,  +1.0e-3),   # Stratosphere (1)
        (32e3,  47e3,  +2.8e-3),   # Stratosphere (2)
        (47e3,  51e3,   0.0),      # Stratopause region (isothermal)
        (51e3,  71e3,  -2.8e-3),   # Mesosphere (1)
        (71e3,  86e3,  -2.0e-3),   # Mesosphere (2)
    ]

    # prepare outputs:
    T = np.empty_like(h_m)
    p = np.empty_like(h_m)

    # we compute layer by layer, keeping base conditions (Tb, pb) consistent.
    Tb = T0
    pb = p0
    hb_prev = 0.0

    # helper: advance base conditions to a new base altitude by integrating through layers;
    # in this implementation we do it naturally while looping over layers:

    for hb, ht, L in layers:
        # If we skipped something (should not happen), ensure consistency
        if hb != hb_prev:
            raise RuntimeError("Layer bases are not contiguous. Check layer list.")

        # Mask of points in this layer
        mask = (h_m >= hb) & (h_m <= ht)
        h = h_m[mask]

        # temperature profile in this layer:
        if L == 0.0:
            T_layer = Tb * np.ones_like(h)
            # isothermal barometric formula:
            p_layer = pb * np.exp(-g0 * (h - hb) / (R * Tb))
        else:
            T_layer = Tb + L * (h - hb)
            # power law for constant lapse rate:
            p_layer = pb * (T_layer / Tb) ** (-g0 / (R * L))

        T[mask] = T_layer
        p[mask] = p_layer

        # update base conditions for next layer at its base (= current layer top):
        # evaluate at ht:
        if L == 0.0:
            Tt = Tb
            pt = pb * np.exp(-g0 * (ht - hb) / (R * Tb))
        else:
            Tt = Tb + L * (ht - hb)
            pt = pb * (Tt / Tb) ** (-g0 / (R * L))

        Tb, pb = Tt, pt
        hb_prev = ht

    rho = p / (R * T)
    return T, p, rho
# %% MAIN SCRIPT

# altitude grid:
h_km = np.linspace(0.0, 86.0, 800)
h_m = 1e3 * h_km

# compute profiles:
T, p, rho = standard_atmosphere_1d(h_m)

# layer boundaries to annotate (km):
bounds_km = np.array([0, 11, 20, 32, 47, 51, 71, 86], dtype=float)

# a convenient reference height marker (example: 8 km):
h_ref_km = 8.0


# plots:
fig, axs = plt.subplots(1, 3, figsize=(7, 7), sharey=True)

# temperature plot:
axs[0].plot(T, h_km, lw=3, c="tab:blue")
axs[0].set_xlabel(r"$T(h)$ [K]")
axs[0].set_ylabel("Altitude [km]")
axs[0].set_xlim(170, 300)
axs[0].set_yticks(bounds_km)
axs[0].set_title("Temperature")

# pressure plot: (convert to hPa for better readability)
axs[1].plot(p * 1e-2, h_km, lw=3, c="tab:orange")
axs[1].set_xlabel(r"$p(h)$ [hPa]")
axs[1].set_xlim(-10, 1013.25)
axs[1].set_title("Pressure")

# density plot:
axs[2].plot(rho, h_km, lw=3, c="tab:green")
axs[2].set_xlabel(r"$\rho(h)$ [kg/m$^3$]")
axs[2].set_xlim(-0.025, 1.25)
axs[2].set_title("Density")

# boundary lines and simple region labels:
for ax in axs:
    for b in bounds_km:
        ax.axhline(b, linestyle="--", linewidth=0.8, color="k", alpha=0.45)
    ax.grid(True, alpha=0.25)

# labels on the left panel (kept compact):
axs[0].text(170, 2,  "Troposphere", fontsize=9, va="center")
axs[0].text(170, 11, "Tropopause", fontsize=9, va="bottom")
axs[0].text(170, 22, "Stratosphere (1)", fontsize=9, va="center")
axs[0].text(170, 34, "Stratosphere (2)", fontsize=9, va="center")
axs[0].text(170, 47, "Stratopause", fontsize=9, va="bottom")
axs[0].text(170, 53, "Mesosphere (1)", fontsize=9, va="center")
axs[0].text(170, 72, "Mesosphere (2)", fontsize=9, va="bottom")

# add super-title:
fig.suptitle("1D Hydrostatic Atmosphere Profiles\n(1976 U.S. Standard Atmosphere)", fontsize=14)

fig.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "hydrostatic_atmosphere_1D_profiles.png"), dpi=300)
plt.close(fig)
# %% END