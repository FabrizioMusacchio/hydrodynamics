"""
1D hydrostatic ocean water column (static reference profiles)

This script computes idealized, physically plausible vertical profiles in the ocean:
  T(z)     in situ temperature [degC]
  S(z)     practical salinity [psu]
  p(z)     pressure [dbar]
  rho(z)   in situ density [kg/m^3]
and produces three plots analogous to a simple 1D standard atmosphere figure:
  temperature vs depth
  pressure vs depth
  density vs depth

Coordinate convention
  z is depth in meters, positive downward, z = 0 at the sea surface.

Physics
  Hydrostatic balance:
    dp/dz = rho(z) g

  Equation of state for seawater:
    rho = rho(S, T, p)

Implementation strategy
  * Define idealized T(z) and S(z) resembling a mixed layer, thermocline, deep ocean.
  * Compute pressure p(z) by integrating dp/dz using a fixed point iteration:
      p_{k+1}(z) = ∫_0^z rho(S, T, p_k) g dz
    We carry pressure in Pa internally and convert to dbar for oceanographic convention.

Thermodynamics
  If the package "gsw" (TEOS-10 Gibbs SeaWater) is available, we compute density using TEOS-10.
  Otherwise, we fall back to a simple linearized equation of state with a compressibility term.
  The fallback is for didactics and plotting only and is not meant for high accuracy work.

Notes
  * This is not an ocean circulation model. It is a static 1D reference column.
  * If you want Brunt–Väisälä frequency N^2(z), you can add it easily once rho(z) is computed.
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

# TEOS-10 (gsw):
import gsw

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% CONSTANTS AND BASE CONDITIONS

# constants:
g0 = 9.80665          # m/s^2
p_ref = 101325.0      # Pa, reference pressure (not crucial here)
PA_PER_DBAR = 1e4     # 1 dbar = 10^4 Pa (definition)


# output folder:
RESULTS_PATH = "hydrostatic_ocean_1D_figures"
os.makedirs(RESULTS_PATH, exist_ok=True)
# %% FUNCTIONS

# Idealized ocean profiles:
def idealized_temperature_Salinity(z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct an idealized T(z) and S(z) profile.

    We use smooth transitions via tanh to mimic:
      * a mixed layer (nearly uniform T, S near surface)
      * a thermocline and halocline
      * deep ocean values

    Parameters
    ----------
    z_m : np.ndarray
        Depth in meters (positive down).

    Returns
    -------
    T_C : np.ndarray
        In situ temperature in degC.
    SP : np.ndarray
        Practical salinity in psu.
    """
    z = np.asarray(z_m, dtype=float)

    # mixed layer thickness and thermocline center and thickness:
    z_ml = 80.0          # m
    z_tc = 600.0         # m
    w_tc = 250.0         # m, transition width

    # surface and deep temperatures:
    T_surface = 20.0     # degC
    T_deep = 2.0         # degC

    # build a smooth decay from surface to deep:
    # f(z) transitions from ~0 near surface to ~1 in deep ocean
    fT = 0.5 * (1.0 + np.tanh((z - z_tc) / w_tc))
    T_C = T_surface * (1.0 - fT) + T_deep * fT

    # enforce a well mixed near surface: flatten T above z_ml;
    # use another smooth switch so there is no sharp kink:
    fML = 0.5 * (1.0 + np.tanh((z - z_ml) / 20.0))
    T_C = (1.0 - fML) * T_surface + fML * T_C

    # salinity: mild surface value, slightly higher at depth, with gentle structure:
    S_surface = 34.6
    S_deep = 34.9
    fS = 0.5 * (1.0 + np.tanh((z - 300.0) / 200.0))
    SP = S_surface * (1.0 - fS) + S_deep * fS

    # add a subtle mid depth maximum (e.g., to mimic an intermediate water mass):
    SP += 0.05 * np.exp(-0.5 * ((z - 1000.0) / 400.0) ** 2)

    return T_C, SP

# density models:
def rho_teos10(SP: np.ndarray, T_C: np.ndarray, p_dbar: np.ndarray,
               lat: float = 30.0, lon: float = -40.0) -> np.ndarray:
    """
    Compute in situ density using TEOS-10 via gsw.

    We need an approximate location to convert SP, T, p to Absolute Salinity and Conservative Temperature.
    This reference location is a convention. It affects SA conversion slightly.
    For an idealized column, this is acceptable.

    Parameters
    ----------
    SP : np.ndarray
        Practical salinity [psu].
    T_C : np.ndarray
        In situ temperature [degC].
    p_dbar : np.ndarray
        Sea pressure [dbar] (approximately equals absolute pressure minus 1 atm, but gsw expects sea pressure).
    lat, lon : float
        Reference location for SP -> SA conversion.

    Returns
    -------
    rho : np.ndarray
        In situ density [kg/m^3].
    """
    if gsw is None:
        raise RuntimeError("gsw is not available")

    SP = np.asarray(SP, dtype=float)
    T_C = np.asarray(T_C, dtype=float)
    p_dbar = np.asarray(p_dbar, dtype=float)

    SA = gsw.SA_from_SP(SP, p_dbar, lon, lat)     # g/kg
    CT = gsw.CT_from_t(SA, T_C, p_dbar)           # degC
    rho = gsw.rho(SA, CT, p_dbar)                 # kg/m^3
    return rho


def rho_linearized(SP: np.ndarray, T_C: np.ndarray, p_Pa: np.ndarray) -> np.ndarray:
    """
    Simple fallback density: linear thermal expansion, haline contraction, and small compressibility.

    rho ≈ rho0 [1 - alpha (T - T0) + beta (S - S0)] * [1 + kappa (p - p0)]

    This is not TEOS-10. It is only a qualitative approximation.

    Parameters
    ----------
    SP : np.ndarray
        Practical salinity [psu].
    T_C : np.ndarray
        Temperature [degC].
    p_Pa : np.ndarray
        Pressure [Pa].

    Returns
    -------
    rho : np.ndarray
        Density [kg/m^3].
    """
    rho0 = 1027.0
    T0 = 10.0
    S0 = 35.0

    # typical magnitudes (order of magnitude):
    alpha = 2.0e-4     # 1/K
    beta = 7.6e-4      # 1/psu

    # effective compressibility:
    # seawater bulk modulus ~ 2.2 GPa, so kappa ~ 1/K ~ 4.5e-10 1/Pa
    kappa = 4.5e-10

    rho_TS = rho0 * (1.0 - alpha * (T_C - T0) + beta * (SP - S0))
    rho = rho_TS * (1.0 + kappa * (p_Pa - p_ref))
    return rho

# hydrostatic integration:
def hydrostatic_ocean_column(
    z_m: np.ndarray,
    T_C: np.ndarray,
    SP: np.ndarray,
    use_gsw: bool = True,
    max_iter: int = 30,
    tol_rel: float = 1e-7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hydrostatic pressure and density for a 1D ocean column.

    We solve dp/dz = rho g with rho depending on p via equation of state.
    A fixed point iteration is used because p appears inside rho(S, T, p).

    Parameters
    ----------
    z_m : np.ndarray
        Depth grid [m], increasing from 0.
    T_C : np.ndarray
        Temperature profile [degC].
    SP : np.ndarray
        Salinity profile [psu].
    use_gsw : bool
        If True and gsw is available, use TEOS-10.
        Otherwise use the linearized fallback.
    max_iter : int
        Maximum number of fixed point iterations.
    tol_rel : float
        Relative tolerance on pressure for convergence.

    Returns
    -------
    p_dbar : np.ndarray
        Pressure [dbar].
    rho : np.ndarray
        Density [kg/m^3].
    p_Pa : np.ndarray
        Pressure [Pa].
    """
    z = np.asarray(z_m, dtype=float)
    if np.any(z < 0):
        raise ValueError("Depth z must be >= 0 m.")
    if np.any(np.diff(z) <= 0):
        raise ValueError("Depth grid must be strictly increasing.")

    T_C = np.asarray(T_C, dtype=float)
    SP = np.asarray(SP, dtype=float)
    if T_C.shape != z.shape or SP.shape != z.shape:
        raise ValueError("z, T_C, and SP must have the same shape.")

    # initial pressure guess: assume constant density 1025 kg/m^3:
    rho_guess = 1025.0
    p_Pa = rho_guess * g0 * z
    p_old = p_Pa.copy()

    for _ in range(max_iter):
        p_dbar = p_Pa / PA_PER_DBAR

        if use_gsw and (gsw is not None):
            rho = rho_teos10(SP, T_C, p_dbar)
        else:
            rho = rho_linearized(SP, T_C, p_Pa)

        # Integrate dp/dz = rho g with trapezoidal rule
        dp_dz = rho * g0
        p_new = np.zeros_like(z)
        p_new[1:] = np.cumsum(0.5 * (dp_dz[1:] + dp_dz[:-1]) * np.diff(z))

        # Convergence check
        denom = np.maximum(np.abs(p_new), 1.0)
        rel = np.max(np.abs(p_new - p_old) / denom)
        p_Pa = p_new
        if rel < tol_rel:
            break
        p_old = p_Pa.copy()

    # final rho on converged pressure:
    p_dbar = p_Pa / PA_PER_DBAR
    if use_gsw and (gsw is not None):
        rho = rho_teos10(SP, T_C, p_dbar)
    else:
        rho = rho_linearized(SP, T_C, p_Pa)

    return p_dbar, rho, p_Pa
# %% MAIN SCRIPT

# depth grid:
z_max = 5000.0
z_m = np.linspace(0.0, z_max, 900)

# idealized T and S:
T_C, SP = idealized_temperature_Salinity(z_m)

# Hydrostatic column
use_gsw = True
p_dbar, rho, _p_Pa = hydrostatic_ocean_column(z_m, T_C, SP, use_gsw=use_gsw)

# some plot settings:
z_km = z_m * 1e-3 # convert to km

# depth ticks (km):
depth_ticks_km = np.array([0, 0.1, 0.5, 1, 2, 3, 4, 5], dtype=float)


# plotting:
fig, axs = plt.subplots(1, 4, figsize=(7, 7), sharey=True)

# plot temperature:
axs[0].plot(T_C, z_km, lw=3, c="tab:blue")
axs[0].set_xlabel(r"$T(z)$ [°C]")
axs[0].set_ylabel("Depth [km]")
axs[0].set_title("Temperature")
axs[0].set_ylim(depth_ticks_km[-1], depth_ticks_km[0])

# plot pressure:
axs[1].plot(p_dbar, z_km, lw=3, c="tab:orange")
axs[1].set_xlabel(r"$p(z)$ [dbar]")
axs[1].set_title("Pressure")

# plot density:
axs[2].plot(rho, z_km, lw=3, c="tab:green")
axs[2].set_xlabel(r"$\rho(z)$ [kg/m$^3$]")
axs[2].set_title("Density")

# plot salinity:
axs[3].plot(SP, z_km, lw=3, c="tab:purple")
axs[3].set_xlabel(r"$S(z)$ [psu]")
axs[3].set_title("Salinity")
# optional, but typically nice for readability:
axs[3].set_xlim(np.min(SP) - 0.1, np.max(SP) + 0.1)

# annotate typical layers on the left panel:
axs[0].text(np.min(T_C), 0.15, "Mixed layer", fontsize=10, va="center")
axs[0].text(np.max(T_C), 0.8, "Thermocline", fontsize=10, va="center", ha="right")
axs[0].text(np.max(T_C), 3.0, "Deep ocean", fontsize=10, va="center", ha="right")

# depth axis direction: ocean increases downward, so invert y-axis:
for ax in axs:
    ax.set_yticks(depth_ticks_km)
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()

# title:
thermo_label = "TEOS-10 (gsw)"
fig.suptitle(f"1D Hydrostatic Ocean Column Profiles\n(idealized T and S, {thermo_label})", fontsize=14)

fig.tight_layout()

if use_gsw and (gsw is not None):
    outpath = os.path.join(RESULTS_PATH, f"hydrostatic_ocean_1D_profiles_teos10.png")
else:
    outpath = os.path.join(RESULTS_PATH, f"hydrostatic_ocean_1D_profiles_linearized.png")
plt.savefig(outpath, dpi=300)
plt.close(fig)
# %% END