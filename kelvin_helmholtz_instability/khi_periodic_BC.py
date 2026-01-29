"""
2D Kelvin–Helmholtz instability in an incompressible shear flow
(spatially developing setup on a periodic domain, low to moderate Reynolds number).

This script solves the two-dimensional incompressible Navier–Stokes equations
in vorticity–streamfunction formulation,

    ∂t ω + u · ∇ω = ν ∇²ω + F(x, y, t),
    ∇²ψ = −ω,
    u = (u, v) = (∂y ψ, −∂x ψ),

using a pseudospectral method with Fourier discretization in both spatial
directions and explicit second-order Runge–Kutta time stepping.

Physical setup
---------------
A planar shear flow is prescribed as the base state,

    u0(y) = U_mean + U0 * tanh((y − y0) / δ),

representing two parallel streams separated by a finite-thickness shear layer
of width δ. The parameter U_mean introduces a net advection in the positive
x-direction, while U0 controls the shear strength. The corresponding base-state
vorticity is given by ω0(y) = −∂y u0(y).

Kelvin–Helmholtz instability develops from small transverse perturbations
imposed near the left side of the domain (x ≈ 0). These perturbations are
implemented as localized initial vorticity fluctuations and, optionally, as a
time-dependent vorticity forcing term F(x, y, t) with compact support in x.
This forcing mimics small inlet disturbances of the tangential velocity.

Numerical method
----------------
Spatial derivatives are evaluated spectrally using fast Fourier transforms.
Nonlinear advection terms are computed in physical space and dealiased using
the 2/3 rule to prevent aliasing-driven numerical instabilities. Viscous
diffusion is treated explicitly. The time step is adjusted dynamically using
a CFL condition based on the instantaneous velocity field.

Boundary conditions and their implications
-------------------------------------------
The computational domain is periodic in both x and y directions. As a
consequence, there are no physical inlet or outlet boundaries. Structures that
are advected out of the domain in x re-enter on the opposite side. The localized
perturbation near x ≈ 0 therefore acts as a *source region* rather than a true
inlet boundary.

Because of the periodicity in x, vortices and coherent structures may appear
anywhere in the domain, including near the right edge, without implying a
physical boundary interaction. The simulation should thus be interpreted as a
shear layer evolving in a repeating domain, not as a strict inlet–outlet flow.
This choice simplifies the Poisson solve for the streamfunction and allows an
efficient spectral implementation, but it precludes a strictly spatially
developing Kelvin–Helmholtz configuration with non-periodic outflow conditions.

Scope and limitations
---------------------
The model is intended as a physically consistent and numerically robust
demonstration of Kelvin–Helmholtz roll-up and vortex dynamics in a shear flow at
low to moderate Reynolds numbers. While the imposed perturbations are localized
in x and the mean flow introduces downstream advection, the periodic boundary
conditions imply that the evolution is only *quasi* spatially developing.
For studies requiring true inlet–outlet dynamics, non-periodic boundary
conditions and a different Poisson solver would be required.

author: Fabrizio Musacchio
date: Jan 2021 / Jan 2026
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% FUNCTIONS

# functions for spectral derivatives, Poisson solve, RHS evaluation, plotting
def make_wavenumbers(n: int, L: float) -> np.ndarray:
    """
    FFT wavenumbers in rad per length.
    """
    return 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)

# spectral derivative functions:
def spectral_derivative_x(f_hat: np.ndarray, kx: np.ndarray) -> np.ndarray:
    """
    Compute d/dx of a field given its Fourier transform along x and y (2D FFT).
    """
    return np.fft.ifft2(1j * kx[:, None] * f_hat).real
def spectral_derivative_y(f_hat: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Compute d/dy of a field given its Fourier transform along x and y (2D FFT).
    """
    return np.fft.ifft2(1j * ky[None, :] * f_hat).real

# function for spectral Laplacian:
def laplacian_hat(f_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Spectral Laplacian: -(kx^2 + ky^2) * f_hat
    """
    k2 = (kx[:, None] ** 2) + (ky[None, :] ** 2)
    return -(k2) * f_hat

# function to solve Poisson equation for streamfunction:
def poisson_solve_streamfunction(omega: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """
    Solve ∇^2 ψ = -ω in Fourier space on a periodic domain:
    ψ_hat = ω_hat / (kx^2 + ky^2), with zero mode set to 0.
    """
    omega_hat = np.fft.fft2(omega)
    k2 = (kx[:, None] ** 2) + (ky[None, :] ** 2)
    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)

    mask = k2 > 0.0
    psi_hat[mask] = omega_hat[mask] / k2[mask]
    psi_hat[~mask] = 0.0

    psi = np.fft.ifft2(psi_hat).real
    return psi

# function to compute velocity from streamfunction:
def compute_velocity_from_streamfunction(psi: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    u =  dψ/dy, v = -dψ/dx
    """
    psi_hat = np.fft.fft2(psi)
    u = spectral_derivative_y(psi_hat, ky)
    v = -spectral_derivative_x(psi_hat, kx)
    return u, v

# function to create dealiasing mask:
def make_dealias_mask(Nx: int, Ny: int) -> np.ndarray:
    """
    2/3-rule dealiasing mask for 2D FFT grids.
    Keeps modes |k| <= N/3 in each direction.
    """
    kx_idx = np.fft.fftfreq(Nx) * Nx
    ky_idx = np.fft.fftfreq(Ny) * Ny
    kx_max = Nx // 3
    ky_max = Ny // 3
    mask = (np.abs(kx_idx)[:, None] <= kx_max) & (np.abs(ky_idx)[None, :] <= ky_max)
    return mask.astype(float)

# function to compute RHS of vorticity equation:
def rhs_vorticity(omega: np.ndarray, nu: float, kx: np.ndarray, ky: np.ndarray, forcing: np.ndarray,
                  dealias_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Right-hand side: -u·∇ω + ν∇^2 ω + forcing
    """
    psi = poisson_solve_streamfunction(omega, kx, ky)
    u, v = compute_velocity_from_streamfunction(psi, kx, ky)

    omega_hat = np.fft.fft2(omega)
    domega_dx = spectral_derivative_x(omega_hat, kx)
    domega_dy = spectral_derivative_y(omega_hat, ky)

    adv = u * domega_dx + v * domega_dy

    # dealias advection term (important for stability of pseudospectral nonlinear products):
    if dealias_mask is not None:
        adv_hat = np.fft.fft2(adv)
        adv_hat *= dealias_mask
        adv = np.fft.ifft2(adv_hat).real

    lap_omega = np.fft.ifft2(laplacian_hat(omega_hat, kx, ky)).real

    return -adv + nu * lap_omega + forcing

# plotting function to convert figure to RGB array:
def fig_to_rgb_array(fig) -> np.ndarray:
    """
    Convert a Matplotlib figure to an RGB uint8 image array of shape (H, W, 3).
    Works across Matplotlib backends/versions using buffer_rgba().
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # RGBA buffer, length = w*h*4
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))

    # Drop alpha channel
    rgb = buf[:, :, :3].copy()
    return rgb

# function to render a frame of vorticity field:
def render_frame(x, y, omega, t, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    extent = [x.min(), x.max(), y.min(), y.max()]

    im = ax.imshow(omega.T, origin="lower", extent=extent, aspect="auto",
        vmin=vmin, vmax=vmax,cmap="RdBu_r",)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"2D Kelvin-Helmholtz vorticity  t = {t:.3f}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("vorticity ω")

    fig.tight_layout()
    rgb = fig_to_rgb_array(fig)
    plt.close(fig)
    return rgb


# %% MAIN RUN

# domain and numerics:
Nx = 512        # number of grid points in x
Ny = 256        # number of grid points in y
Lx = 300.0      # domain length in x
Ly = 70.0       # domain length in y

x = np.linspace(0.0, Lx, Nx, endpoint=False)
y = np.linspace(0.0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

kx = make_wavenumbers(Nx, Lx)
ky = make_wavenumbers(Ny, Ly)

dx = Lx / Nx    # grid spacing in x
dy = Ly / Ny    # grid spacing in y


# physical parameters:
U0 = 2.5           # shear velocity scale, start with 1.0, then increase for higher Re
Umean = 2.0        # net advection to +x
delta = 1.5        # shear layer thickness scale
nu = 0.012          # viscosity (choose larger for lower Reynolds number)

# time stepping:
dt = 0.05
nsteps = 2200

# CFL control (stability):
cfl = 0.30
dt_max = 0.03
dt_min = 1e-4


# output control:
frame_every = 10
frames_dir = "khi_periodic_BC_frames"
gif_path = "kelvin_helmholtz_2d.gif"
os.makedirs(frames_dir, exist_ok=True)


# base shear profile u0(y):
y0 = 0.5 * Ly
#u0 = U0 * np.tanh((y - y0) / delta)  # 1D in y
u0 = Umean + U0 * np.tanh((y - y0) / delta)
du0_dy = (U0 / delta) * (1.0 / np.cosh((y - y0) / delta) ** 2)
omega0_y = -du0_dy  # ω = ∂v/∂x - ∂u/∂y, base has v=0, u=u(y)

omega = np.repeat(omega0_y[None, :], Nx, axis=0)


# spatially localized initial perturbation near the "inlet" x ~ 0:
eps0 = 0.25
x_loc = 18.0
sigma_x = 14.0
ky_mode = 2.0 * np.pi / Ly * 3.0

inlet_envelope = np.exp(-((X - x_loc) ** 2) / (2.0 * sigma_x ** 2))
omega += eps0 * inlet_envelope * np.sin(ky_mode * Y)


# optional time-dependent inlet forcing F(x,y,t)
# this mimics small perturbations imposed at an inlet on the tangential velocity.
# in vorticity form we inject a localized oscillatory vorticity source near x ~ 0.
forcing_on = True
epsF = 0.10
forcing_sigma_x = 10.0
forcing_x0 = 8.0
forcing_omega_t = 0.35
forcing_ky_mode = 2.0 * np.pi / Ly * 5.0

forcing_x_env = np.exp(-((X - forcing_x0) ** 2) / (2.0 * forcing_sigma_x ** 2))

# determine a stable plotting range from the initial field
q = np.quantile(np.abs(omega), 0.995)
vmin, vmax = -q, q


# MAIN time integration (RK2):
# create storage for output frames:
frame_paths = []
frame_idx = 0

# initialize a GIF writer:
writer = imageio.get_writer(gif_path, mode="I", duration=0.05, loop=0)

# dealiasing mask for nonlinear term:
dealias_mask = make_dealias_mask(Nx, Ny)

# set initial time; use an accumulated physical time if dt becomes adaptive:
t = 0.0

try:
    for n in range(nsteps + 1):

        # compute velocity for CFL-based adaptive dt:
        psi = poisson_solve_streamfunction(omega, kx, ky)
        u, v = compute_velocity_from_streamfunction(psi, kx, ky)
        umax = np.max(np.abs(u))
        vmax_loc = np.max(np.abs(v))
        speed_eps = 1e-12

        dt_cfl = cfl * min(dx / (umax + speed_eps), dy / (vmax_loc + speed_eps))
        dt = float(np.clip(dt_cfl, dt_min, dt_max))

        if forcing_on:
            forcing = epsF * forcing_x_env * np.sin(forcing_ky_mode * Y) * np.sin(forcing_omega_t * t)
        else:
            forcing = np.zeros_like(omega)

        # save frame every frame_every steps:
        if (n % frame_every) == 0:
            rgb = render_frame(x, y, omega, t, vmin=vmin, vmax=vmax)

            outpath = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
            imageio.imwrite(outpath, rgb)     # PNG from the exact same pixels
            writer.append_data(rgb)           # GIF frame from RAM

            frame_idx += 1
            print(f"Saved {outpath}")

        # RK2 step:
        k1 = rhs_vorticity(omega, nu, kx, ky, forcing, dealias_mask=dealias_mask)
        omega_tmp = omega + dt * k1

        if forcing_on:
            forcing2 = epsF * forcing_x_env * np.sin(forcing_ky_mode * Y) * np.sin(forcing_omega_t * (t + dt))
        else:
            forcing2 = np.zeros_like(omega)

        k2 = rhs_vorticity(omega_tmp, nu, kx, ky, forcing2, dealias_mask=dealias_mask)
        omega = omega + 0.5 * dt * (k1 + k2)

        if not np.isfinite(omega).all():
            print(f"Non-finite omega encountered at step {n}, t={t:.3f}. Aborting.")
            break

        t += dt

finally:
    writer.close()

print(f"Saved {gif_path}")
# %% END