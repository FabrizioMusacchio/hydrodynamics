# %% IMPORTS
import os
import shutil
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% FUNCTIONS

def make_2d_wavenumbers(nx: int, ny: int, lx: float, ly: float):
    """ 
    Function to create 2D wavenumber grids kx, ky and squared wavenumber k2.
    """
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")
    k2 = kx**2 + ky**2
    return kx, ky, k2


def dealias_mask(nx: int, ny: int):
    """ 
    2/3 rule in Fourier space, rectangular mask.
    """
    kx_cut = nx // 3
    ky_cut = ny // 3
    kx_idx = np.fft.fftfreq(nx) * nx
    ky_idx = np.fft.fftfreq(ny) * ny
    kx_i, ky_i = np.meshgrid(kx_idx, ky_idx, indexing="xy")
    mask = (np.abs(kx_i) <= kx_cut) & (np.abs(ky_i) <= ky_cut)
    return mask.astype(float)


def omega_to_rgb_autoscale(omega: np.ndarray, cmap_name: str = "RdBu_r", perc: float = 99.5):
    """ 
    Convert vorticity field to RGB image with autoscaling based on percentile.
    """
    vlim = np.percentile(np.abs(omega), perc)
    vlim = max(vlim, 1e-12)
    cmap = cm.get_cmap(cmap_name)
    x = np.clip(omega / vlim, -1.0, 1.0)
    x01 = 0.5 * (x + 1.0)
    rgba = cmap(x01)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb, vlim


def save_png(rgb: np.ndarray, path: str):
    """ 
    Save RGB image as PNG without axes or borders.
    """
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    fig.savefig(path, dpi=150)
    plt.close(fig)

def ring_forcing_realspace_hat(nx, ny, lx, ly, kf: float, width: float, nmodes: int, amp: float, rng):
    """
    Generate an isotropic stochastic real-space forcing as a sum of plane waves
    and return its 2D Fourier transform.

    The forcing is constructed as the sum of `nmodes` plane waves with random
    directions and phases, where each wavevector magnitude |k| is sampled from the
    interval [kf*(1-width), kf*(1+width)]. The resulting real-space field is
    normalized to unit standard deviation and scaled by `amp` before computing
    the FFT. The mean (zero wavenumber) component is set to zero in the output.

    Parameters
    ----------
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    lx : float
        Domain length in the x direction.
    ly : float
        Domain length in the y direction.
    kf : float
        Forcing wavenumber in "FFT mode" units (e.g. kf=12 ≈ 12 periods over 2π).
    width : float
        Relative half-width of the ring in k-space (sampling interval is
        [kf*(1-width), kf*(1+width)]).
    nmodes : int
        Number of random plane-wave modes to sum.
    amp : float
        Desired amplitude (standard deviation) of the forcing after normalization.
    rng : object
        Random number generator providing `uniform(low, high)`; used for sampling
        wavevector magnitudes, directions, and phases.

    Returns
    -------
    f_hat : ndarray of complex, shape (ny, nx)
        2D Fourier transform of the generated real-space forcing. The zero
        wavenumber component is explicitly set to 0.
    """
    x = (lx / nx) * np.arange(nx)
    y = (ly / ny) * np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    f = np.zeros((ny, nx), dtype=np.float64)

    kmin = (1.0 - width) * kf
    kmax = (1.0 + width) * kf

    for _ in range(nmodes):
        kk = rng.uniform(kmin, kmax)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        ph = rng.uniform(0.0, 2.0 * np.pi)

        kx = kk * np.cos(theta)
        ky = kk * np.sin(theta)

        f += np.cos(kx * xx + ky * yy + ph)

    f /= max(np.std(f), 1e-12)
    f *= amp

    f_hat = np.fft.fft2(f)
    f_hat[0, 0] = 0.0 + 0.0j
    return f_hat

def rhs_vorticity_and_uv(omega_hat, kx, ky, k2, nu: float, alpha: float, dealias: np.ndarray, f_hat):
    """
    Compute the right-hand side of the vorticity equation in Fourier space and
    return the real-space velocity components for CFL diagnostics.

    Parameters
    ----------
    omega_hat : ndarray (complex)
        Vorticity in Fourier space (2D).
    kx, ky : ndarray (real)
        2D arrays of wavenumber components in x and y directions.
    k2 : ndarray (real)
        Squared wavenumber array (kx**2 + ky**2).
    nu : float
        Kinematic viscosity (diffusion coefficient).
    alpha : float
        Linear drag (friction) coefficient.
    dealias : ndarray or boolean mask
        Dealiasing mask applied to the nonlinear (advective) term in spectral space.
    f_hat : ndarray (complex)
        External forcing in Fourier space.

    Returns
    -------
    rhs_hat : ndarray (complex)
        Right-hand side of the vorticity equation in Fourier space:
        - advective_term (dealiased) + diffusion + linear drag + forcing.
    u, v : ndarray (real)
        Real-space velocity components computed from the streamfunction:
        u = ∂ψ/∂y, v = -∂ψ/∂x. Provided for CFL diagnostics.

    Notes
    -----
    - The streamfunction ψ̂ is obtained by inversion ψ̂ = -ω̂ / k2 with ψ̂[0,0] set to 0.
    - Nonlinear advection is computed in physical space and then transformed back
      to spectral space; the provided dealias mask is applied to the nonlinear term.
    """
    psi_hat = np.zeros_like(omega_hat)
    psi_hat[k2 != 0] = -omega_hat[k2 != 0] / k2[k2 != 0]
    psi_hat[0, 0] = 0.0 + 0.0j

    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real

    domega_dx = np.fft.ifft2(1j * kx * omega_hat).real
    domega_dy = np.fft.ifft2(1j * ky * omega_hat).real

    adv = u * domega_dx + v * domega_dy
    adv_hat = np.fft.fft2(adv) * dealias

    diff_hat = -nu * k2 * omega_hat
    drag_hat = -alpha * omega_hat

    rhs_hat = -adv_hat + diff_hat + drag_hat + f_hat
    return rhs_hat, u, v

def isotropic_energy_spectrum_from_omega_hat(omega_hat, kx, ky, k2, lx, ly, nbins=120):
    """
    Compute the isotropically averaged energy spectrum E(k) from omega_hat.

    Returns:
        k_centers: bin centers (float)
        E_k: isotropically averaged spectrum (float)

    Notes:
    * Normalization depends on FFT conventions. For cascade analysis the shape is most important.
    * k here is the physical wavenumber (rad/length), since kx, ky are constructed in rad/length.
    """
    ny, nx = omega_hat.shape

    # streamfunction:
    psi_hat = np.zeros_like(omega_hat)
    psi_hat[k2 != 0] = -omega_hat[k2 != 0] / k2[k2 != 0]
    psi_hat[0, 0] = 0.0 + 0.0j

    # velocity components (modes) in Fourier space:
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    # FFT normalization: numpy ifft has a 1/(N) factor, fft does not
    # FFor Parseval consistency we scale here with (1/(nx*ny))^2.
    # The exact factor is not critical for the spectral shape.
    norm = (1.0 / (nx * ny))**2
    e_mode = 0.5 * norm * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    k_mag = np.sqrt(k2).ravel()
    e_flat = e_mode.ravel()

    # exclude k=0:
    mask = k_mag > 0
    k_mag = k_mag[mask]
    e_flat = e_flat[mask]

    # binning:
    k_min = k_mag.min()
    k_max = k_mag.max()

    bins = np.linspace(k_min, k_max, nbins + 1)
    #bins = np.logspace(np.log10(k_min), np.log10(k_max), nbins + 1)
    which = np.digitize(k_mag, bins) - 1
    which = np.clip(which, 0, nbins - 1)

    # ring-sums per bin:
    Ek_sum = np.bincount(which, weights=e_flat, minlength=nbins)

    # division by bin width -> E(k) as density:
    dk = bins[1:] - bins[:-1]
    E_k = Ek_sum / np.maximum(dk, 1e-12)

    k_centers = 0.5 * (bins[:-1] + bins[1:])
    return k_centers, E_k

def fit_power_law_slope(k, E, k_lo, k_hi, eps=1e-30):
    """
    Fit log E = a + m log k in the range in [k_lo, k_hi].
    Returns m (slope), a (intercept), and the number of points used in the fit.
    """
    k = np.asarray(k)
    E = np.asarray(E)

    mask = (k >= k_lo) & (k <= k_hi) & np.isfinite(k) & np.isfinite(E) & (E > eps)
    k_use = k[mask]
    E_use = E[mask]

    if k_use.size < 5:
        return np.nan, np.nan, int(k_use.size)

    x = np.log(k_use)
    y = np.log(E_use)

    # linear regression of first order
    m, a = np.polyfit(x, y, 1)  # y ~ m x + a
    return float(m), float(a), int(k_use.size)

def spectrum_frame_png(
    k_centers,
    E_k,
    path,
    kf=None,
    title=None,
    # Fit options
    fit_left=True,
    fit_right=True,
    left_frac=(0.45, 1.05),    # Fit range relative to kf: [0.6*kf, 0.9*kf]
    right_frac=(1.5, 3.5),   # Fit range relative to kf: [1.2*kf, 4.0*kf]
    right_cap_frac=0.6,      # additionally: k_hi <= right_cap_frac * k_max
    eps=1e-30):
    """
    Save spectrum as log-log plot and optionally fit exponents.

    Notes:
    * Fit ranges are by default chosen relative to kf.
    * If kf is None, no fits are performed.
    """
    k_centers = np.asarray(k_centers)
    E_k = np.asarray(E_k)

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.gca()

    ax.loglog(k_centers, np.maximum(E_k, eps), linewidth=2)

    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")

    if title is not None:
        ax.set_title(title)

    if kf is not None:
        # plot forcing wavenumber line:
        ax.axvline(kf, linestyle="--")
        # annotate:
        ax.text(
            kf * 1.05, ax.get_ylim()[1] * 0.8,
            f"$k_f={kf:.1f}$",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="left",
        )
        
        # plot reference -5/3 left:
        k_ref = np.linspace(0.5*kf, 0.9*kf, 50)
        # skaliere an einen Punkt der Kurve
        k0 = k_ref[0]
        E0 = np.interp(k0, k_centers, E_k)
        ax.loglog(k_ref, E0 * (k_ref / k0)**(-5/3), linestyle=":", linewidth=2,
                    c="tab:cyan", label="-5/3 ref.")

        # plot reference -3 right:
        k_ref2 = np.linspace(1.2*kf, 2.0*kf, 50)
        k0 = k_ref2[0]
        E0 = np.interp(k0, k_centers, E_k)
        ax.loglog(k_ref2, E0 * (k_ref2 / k0)**(-3), linestyle=":", linewidth=2,
                    c="tab:red", label="-3 ref.")
        
        k_max = float(np.max(k_centers))

        annotations = []

        # left fit (inverse energy cascade range):
        if fit_left:
            k_lo = left_frac[0] * kf
            k_hi = left_frac[1] * kf
            mL, aL, nL = fit_power_law_slope(k_centers, E_k, k_lo, k_hi, eps=eps)
            if np.isfinite(mL):
                annotations.append(f"fit k∈[{k_lo:.1f},{k_hi:.1f}]:\nslope={mL:.2f} (N={nL})")

                # draw fit line:
                k_line = np.linspace(k_lo, k_hi, 200)
                E_line = np.exp(aL) * (k_line ** mL)
                ax.loglog(k_line, E_line, linewidth=2, c="tab:green")

        # right fit (direct enstrophy cascade range):
        if fit_right:
            k_lo = right_frac[0] * kf
            k_hi = min(right_frac[1] * kf, right_cap_frac * k_max)
            mR, aR, nR = fit_power_law_slope(k_centers, E_k, k_lo, k_hi, eps=eps)
            if np.isfinite(mR):
                annotations.append(f"fit k∈[{k_lo:.1f},{k_hi:.1f}]:\nslope={mR:.2f} (N={nR})")

                k_line = np.linspace(k_lo, k_hi, 200)
                E_line = np.exp(aR) * (k_line ** mR)
                ax.loglog(k_line, E_line, linewidth=2, c="tab:orange")

        if len(annotations) > 0:
            ax.text(
                0.02, 0.02,
                "\n".join(annotations),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="left",
                #bbox=dict(boxstyle="round", alpha=0.8)
            )
            
    #ax.set_ylim(bottom=eps, top=10**2)
    ax.set_ylim(bottom=10**(-28), top=10**2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def simulate_2d_cascade(
    nx=256, ny=256,
    lx=2*np.pi, ly=2*np.pi,
    nu=5e-4,
    alpha=1e-2,
    dt0=2e-3,
    cfl=0.25,
    nsteps=20000,
    frame_every=20,
    seed=1,
    forcing_kf=12.0,
    forcing_width=0.20,
    forcing_nmodes=32,
    forcing_amp=200.0,
    out_dir="frames_2d_cascade",
    gif_path="cascade_2d.gif",
    gif_path_spec="cascade_2d_spectrum.gif",
    gif_fps=30,
    clear_out_dir=True,
    left_frac=(0.45, 1.05),    # Fit range relative to kf: [0.6*kf, 0.9*kf]
    right_frac=(1.5, 3.5),   # Fit range relative to kf: [1.2*kf, 4.0*kf]
    ):
    
    # add forcing_kf to the out_dir name to distinguish different runs:
    out_dir = f"{out_dir}_kf{forcing_kf}"
    out_dir_frames = os.path.join(out_dir, "frames")
    gif_path = f"{gif_path}_kf{forcing_kf}.gif"
    gif_path_spec = f"{gif_path_spec}_kf{forcing_kf}.gif"
    
    if clear_out_dir and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_frames, exist_ok=True)
    
    energy_dir = os.path.join(out_dir, "energy")
    os.makedirs(energy_dir, exist_ok=True)

    frames_spec = []   # RGB frames of spectrum plots for GIF

    rng = np.random.default_rng(seed)

    kx, ky, k2 = make_2d_wavenumbers(nx, ny, lx, ly)
    dealias = dealias_mask(nx, ny)

    omega0 = 0.1 * rng.standard_normal((ny, nx))
    omega_hat = np.fft.fft2(omega0) * dealias

    dx = lx / nx
    dy = ly / ny

    frames_rgb = []
    nframes = 0
    dt = float(dt0)

    for step in range(nsteps + 1):
        # step = 0
        f_hat = ring_forcing_realspace_hat(
            nx, ny, lx, ly,
            kf=forcing_kf, width=forcing_width,
            nmodes=forcing_nmodes,
            amp=forcing_amp,
            rng=rng) * dealias

        # RK2 with adaptive CFL:
        k1, u, v = rhs_vorticity_and_uv(omega_hat, kx, ky, k2, nu, alpha, dealias, f_hat)
        speed_max = float(np.max(np.sqrt(u*u + v*v)))
        dt_cfl = cfl * min(dx, dy) / max(speed_max, 1e-12)

        # limit dt to prevent jumps
        dt = min(dt, dt_cfl)
        dt = max(dt, 1e-6)

        omega_hat_mid = (omega_hat + 0.5 * dt * k1) * dealias
        k2_rhs, _, _ = rhs_vorticity_and_uv(omega_hat_mid, kx, ky, k2, nu, alpha, dealias, f_hat)
        omega_hat = (omega_hat + dt * k2_rhs) * dealias

        if step % frame_every == 0:
            omega = np.fft.ifft2(omega_hat).real
            finite = bool(np.isfinite(omega).all())
            om_rms = float(np.sqrt(np.mean(omega**2)))
            om_max = float(np.max(np.abs(omega)))

            print(f"step={step:6d} dt={dt:.3e} dt_cfl={dt_cfl:.3e} finite={finite} omega_rms={om_rms:.3e} omega_max={om_max:.3e}")

            if not finite:
                raise FloatingPointError("omega contains NaN or Inf")

            rgb, vlim = omega_to_rgb_autoscale(omega, perc=99.5)
            png_path = os.path.join(out_dir_frames, f"frame_{nframes:06d}.png")
            save_png(rgb, png_path)
            frames_rgb.append(rgb)


            # calculate and plot energy spectrum:
            k_centers, E_k = isotropic_energy_spectrum_from_omega_hat(omega_hat, kx, ky, k2, lx, ly, nbins=120)

            spec_path = os.path.join(energy_dir, f"spectrum_{nframes:06d}.png")
            spectrum_frame_png(
                k_centers, E_k,
                path=spec_path,
                kf=forcing_kf,
                title=f"Energy spectrum, frame {nframes}",
                fit_left=True,
                fit_right=True,
                left_frac=left_frac,
                right_frac=right_frac,
            )

            # load PNG as RGB frame into RAM to build a GIF at the end:
            spec_img = imageio.imread(spec_path)
            frames_spec.append(spec_img)

            nframes += 1

    imageio.mimsave(gif_path, frames_rgb, fps=gif_fps)
    print(f"Done. Wrote {nframes} frames. GIF: {gif_path}")
    
    
    imageio.mimsave(gif_path_spec, frames_spec, fps=gif_fps)
    print(f"Energy spectrum GIF saved to: {gif_path_spec}")
    
    return frames_rgb

# %% MAIN
simulate_2d_cascade(
    nx=256, ny=256,         # grid points
    lx=2*np.pi, ly=2*np.pi, # domain size
    nu=1e-4,                # viscosity; start with 5e-4 for moderate, 2e-4 for weak diffusion
    alpha=5e-3,             # linear drag (friction); use: 1e-2 for moderate, 5e-3 for weak drag
    dt0=2e-3,               # initial time step
    cfl=0.25,               # CFL number for adaptive time stepping
    nsteps=20000,           # total number of time steps
    frame_every=20,         # save frame every N steps
    seed=1,                 # random seed
    forcing_kf=12.0,         # wavenumber of forcing ring; larger number leads to smaller structures; e.g., use 6 or 12
    forcing_width=0.20,     # relative width of forcing ring
    forcing_nmodes=32,      # number of random modes in forcing
    forcing_amp=200.0,      # amplitude of forcing
    out_dir="frames_2d_cascade", # output directory for frames
    gif_path="cascade_2d.gif",   # output path for GIF
    gif_path_spec="cascade_2d_spectrum.gif", # output path for spectrum GIF
    gif_fps=30,             # frames per second for GIF    
    clear_out_dir=False,      # clear output directory before running
    left_frac=(0.30, 1.50),    # Fit range relative to kf: [0.6*kf, 0.9*kf]
    right_frac=(1.5, 3.5),   # Fit range relative to
)

""" DEBUGGING

    nx=512
    ny=512
    lx=2*np.pi
    ly=2*np.pi
    nu=1e-4
    alpha=5e-3
    dt0=2e-3
    cfl=0.25
    nsteps=20000
    frame_every=20
    seed=1
    forcing_kf=12.0
    forcing_width=0.20
    forcing_nmodes=32
    forcing_amp=200.0
    out_dir="frames_2d_cascade"
    gif_path="cascade_2d.gif"
    gif_path_spec="cascade_2d_spectrum.gif"
    gif_fps=30
    clear_out_dir=False

"""
# %% END    