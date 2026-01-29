"""
2D spatially developing Kelvin Helmholtz instability of a jet entering from the left.
Finite volume style incompressible Navier Stokes solver (projection method) on a staggered MAC grid.

Model
* Incompressible Navier Stokes in primitive variables (u, v, p)
    ∂t u + u ∂x u + v ∂y u = -∂x p + ν (∂xx u + ∂yy u)
    ∂t v + u ∂x v + v ∂y v = -∂y p + ν (∂xx v + ∂yy v)
    ∂x u + ∂y v = 0
* Explicit time stepping for advection and diffusion, then pressure projection.
* Pressure Poisson equation solved by SOR.
* Also advects a passive scalar "dye" C to visualize the jet.
  This is optional but usually gives the most intuitive KH pictures.

Geometry and BCs
* Domain: x in [0, Lx], y in [-Ly/2, Ly/2], uniform grid.
* y direction periodic.
* Inlet at x=0: prescribed u(y), v(y,t) with small perturbations on the tangential component.
* Outlet at x=Lx: zero gradient (convective like) for u, v, C, and p fixed to 0 for gauge.


This is a compact educational implementation. It is not optimized.

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

# shift operators:
def shift_x_plus(a):
    return np.concatenate([a[1:, ...], a[-1:, ...]], axis=0)

def shift_x_minus(a):
    return np.concatenate([a[:1, ...], a[:-1, ...]], axis=0)

def shift_y_plus(a):
    return np.roll(a, -1, axis=-1)

def shift_y_minus(a):
    return np.roll(a, 1, axis=-1)

def shift_y_plus_v(a):
    return np.roll(a, -1, axis=1)

def shift_y_minus_v(a):
    return np.roll(a, 1, axis=1)


# allocation:
def allocate_fields(Nx, Ny):
    p = np.zeros((Nx, Ny), dtype=float)
    u = np.zeros((Nx + 1, Ny), dtype=float)
    v = np.zeros((Nx, Ny + 1), dtype=float)
    C = np.zeros((Nx, Ny), dtype=float)
    return u, v, p, C


# boundary conditions:
def inlet_profiles(t, params):
    U0 = params["U0"]
    y = params["y_centers"]
    y_faces_v = params["y_faces_v"]
    jet_w = params["jet_w"]
    eps = params["eps"]
    f = params["f"]
    T_ramp = params["T_ramp"]

    # smooth ramp
    s = np.clip(t / T_ramp, 0.0, 1.0)
    ramp = s * s * (3.0 - 2.0 * s)  # smoothstep

    u_in = (ramp * U0) * np.exp(-(y / jet_w) ** 2)
    v_in = (ramp * eps * U0) * np.sin(2.0 * np.pi * f * t) * np.exp(-(y_faces_v / jet_w) ** 2)
    C_in = np.exp(-(y / jet_w) ** 2)

    return u_in, v_in, C_in, ramp


def apply_bc(u, v, p, C, t, dt, dx, params):
    """
    y periodic implicit.
    x inlet: Dirichlet for u and v (with ramp)
    x outlet: convective outlet for u, v, C
    pressure gauge at outlet.
    """
    u_in, v_in, C_in, ramp = inlet_profiles(t, params)

    # inlet
    u[0, :] = u_in
    v[0, :] = v_in
    v[0, 0] = v[0, -1]

    C[0, :] = np.maximum(C[0, :], C_in)

    # convective outlet
    # choose a convective speed based on bulk inlet
    Uc = max(params["U0"] * ramp, 1e-6)
    cfl_out = Uc * dt / dx
    cfl_out = min(cfl_out, 0.95)

    # u outlet: u[-1] is boundary face, use u[-2] interior face
    u[-1, :] = u[-1, :] - cfl_out * (u[-1, :] - u[-2, :])

    # v outlet: v[-1] is boundary column of v faces
    v[-1, :] = v[-1, :] - cfl_out * (v[-1, :] - v[-2, :])

    # scalar outlet at centers
    C[-1, :] = C[-1, :] - cfl_out * (C[-1, :] - C[-2, :])

    # pressure gauge
    p[-1, :] = 0.0


# operators: laplacians:
def laplacian_center(phi, dx, dy):
    d2x = (shift_x_plus(phi) - 2.0 * phi + shift_x_minus(phi)) / (dx * dx)
    d2y = (shift_y_plus(phi) - 2.0 * phi + shift_y_minus(phi)) / (dy * dy)
    return d2x + d2y

def laplacian_u(u, dx, dy):
    d2x = (shift_x_plus(u) - 2.0 * u + shift_x_minus(u)) / (dx * dx)
    d2y = (shift_y_plus(u) - 2.0 * u + shift_y_minus(u)) / (dy * dy)
    return d2x + d2y

def laplacian_v(v, dx, dy):
    d2x = (shift_x_plus(v) - 2.0 * v + shift_x_minus(v)) / (dx * dx)
    d2y = (shift_y_plus_v(v) - 2.0 * v + shift_y_minus_v(v)) / (dy * dy)
    return d2x + d2y


# interpolations:
def interp_u_to_center(u):
    return 0.5 * (u[:-1, :] + u[1:, :])

def interp_v_to_center(v):
    return 0.5 * (v[:, :-1] + v[:, 1:])


# advection: first order upwind:
def advect_u(u, v, dx, dy):
    Nx1, Ny = u.shape
    v_u = np.zeros((Nx1, Ny), dtype=float)
    v_u[1:-1, :] = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])
    v_u[0, :] = v_u[1, :]
    v_u[-1, :] = v_u[-2, :]

    dudx_f = (shift_x_plus(u) - u) / dx
    dudx_b = (u - shift_x_minus(u)) / dx
    du_dx = np.where(u >= 0.0, dudx_b, dudx_f)

    dudy_f = (shift_y_plus(u) - u) / dy
    dudy_b = (u - shift_y_minus(u)) / dy
    du_dy = np.where(v_u >= 0.0, dudy_b, dudy_f)

    return u * du_dx + v_u * du_dy


def advect_v(u, v, dx, dy):
    Nx, Ny1 = v.shape
    u_v = np.zeros((Nx, Ny1), dtype=float)
    u_v[:, 1:-1] = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
    u_v[:, 0] = u_v[:, 1]
    u_v[:, -1] = u_v[:, -2]

    dvdx_f = (shift_x_plus(v) - v) / dx
    dvdx_b = (v - shift_x_minus(v)) / dx
    dv_dx = np.where(u_v >= 0.0, dvdx_b, dvdx_f)

    dvdy_f = (shift_y_plus_v(v) - v) / dy
    dvdy_b = (v - shift_y_minus_v(v)) / dy
    dv_dy = np.where(v >= 0.0, dvdy_b, dvdy_f)

    return u_v * dv_dx + v * dv_dy


# divergence and pressure gradients:
def divergence(u, v, dx, dy):
    return (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dy


def grad_p_to_u(p, dx):
    gp = np.zeros((p.shape[0] + 1, p.shape[1]), dtype=float)
    gp[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx
    gp[0, :] = gp[1, :]
    gp[-1, :] = gp[-2, :]
    return gp

def grad_p_to_v(p, dy):
    gp = np.zeros((p.shape[0], p.shape[1] + 1), dtype=float)
    gp[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy
    gp[:, 0] = gp[:, -2]
    gp[:, -1] = gp[:, 1]
    return gp


# Poisson solver: SOR:
def solve_poisson_fft_y(p, rhs, dx, dy):
    """
    Solve ∇² p = rhs on cell centers with periodic y via FFT.

    Discretization
    * y is periodic, so FFT diagonalizes the y Laplacian
    * for each ky mode we solve a 1D Helmholtz system in x:
        (Dxx - k_y^2) p_hat = rhs_hat
      with
        Neumann at inlet (x=0): p[0] = p[1]
        Dirichlet at outlet (x=Lx): p[Nx-1] = 0

    Returns updated p (real).
    """
    Nx, Ny = rhs.shape

    # FFT in y:
    rhs_hat = np.fft.fft(rhs, axis=1)

    # wavenumbers in y for discrete Laplacian eigenvalues
    # for second difference operator in y:
    # λ_y(ky) = (2 cos(ky dy) - 2) / dy^2 = -4 sin^2(ky dy / 2) / dy^2
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    lam_y = -4.0 * (np.sin(0.5 * ky * dy) ** 2) / (dy * dy)  # shape (Ny,)

    # preallocate solution in Fourier space:
    p_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    # tridiagonal coefficients for x operator:
    ax = 1.0 / (dx * dx)   # sub diagonal
    cx = 1.0 / (dx * dx)   # super diagonal
    # main diagonal depends on mode: bx = -2/dx^2 + lam_y[m]
    # because (Dxx + Dyy) p = rhs, and Dyy eigenvalue is lam_y

    # Thomas solver for each mode in y:
    for m in range(Ny):
        b0 = (-2.0 / (dx * dx) + lam_y[m])  # main diagonal value for this mode

        # build diagonals
        a = ax * np.ones(Nx - 1, dtype=np.complex128)
        b = b0 * np.ones(Nx, dtype=np.complex128)
        c = cx * np.ones(Nx - 1, dtype=np.complex128)
        d = rhs_hat[:, m].astype(np.complex128)

        # apply BCs:
        # Neumann at i=0: p0 - p1 = 0
        b[0] = 1.0
        c[0] = -1.0
        d[0] = 0.0

        # Dirichlet at i=Nx-1: p = 0
        a[-1] = 0.0
        b[-1] = 1.0
        d[-1] = 0.0

        # forward elimination:
        for i in range(1, Nx):
            w = a[i - 1] / b[i - 1]
            b[i] = b[i] - w * c[i - 1]
            d[i] = d[i] - w * d[i - 1]

        # back substitution:
        x = np.zeros(Nx, dtype=np.complex128)
        x[-1] = d[-1] / b[-1]
        for i in range(Nx - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

        p_hat[:, m] = x

    # inverse FFT back to real space:
    p_new = np.fft.ifft(p_hat, axis=1).real

    # Gauge fix at outlet:
    p_new[-1, :] = 0.0
    # inlet Neumann consistency:
    p_new[0, :] = p_new[1, :]

    return p_new



# passive scalar:
def advect_scalar(C, u, v, dx, dy):
    uc = interp_u_to_center(u)
    vc = interp_v_to_center(v)

    Cx_f = (shift_x_plus(C) - C) / dx
    Cx_b = (C - shift_x_minus(C)) / dx
    dC_dx = np.where(uc >= 0.0, Cx_b, Cx_f)

    Cy_f = (shift_y_plus(C) - C) / dy
    Cy_b = (C - shift_y_minus(C)) / dy
    dC_dy = np.where(vc >= 0.0, Cy_b, Cy_f)

    return uc * dC_dx + vc * dC_dy



# diagnostics and plotting:
def compute_vorticity(u, v, dx, dy):
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    dv_dx = (shift_x_plus(v_mid) - shift_x_minus(v_mid)) / (2.0 * dx)
    u_mid = 0.5 * (u[:-1, :] + u[1:, :])
    du_dy = (shift_y_plus(u_mid) - shift_y_minus(u_mid)) / (2.0 * dy)
    return dv_dx - du_dy


def save_frame(frame_path, x_centers, y_centers, field, title, vmin=None, vmax=None,
               ylim=None):
    """ 
    Save a field plot to file.
    
    Parameters
    ----------
    field : (Nx, Ny) array, field on cell centers
    title : str, plot title
    frame_path : str, path to save the plot
    vmin, vmax : float, optional, color limits
    ylim : tuple, optional, y-axis limits
    """
    plt.figure(figsize=(10, 5), dpi=140)
    extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
    plt.imshow(field.T, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax,
               cmap="RdBu_r")
    plt.xlabel("x")
    plt.ylabel("y")
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close()

def save_overlay_frame(frame_path, x_centers, y_centers, omega, C,
                       title,
                       omega_percentile=99.0,
                       C_levels=10,
                       ylim=None):
    """
    Plot vorticity ω as background and overlay passive scalar C as contour lines.
    This makes it easy to see whether the billows in C sit on the shear layers
    where |ω| is large.

    Parameters
    ----------
    omega : (Nx, Ny) array, vorticity on cell centers
    C     : (Nx, Ny) array, passive scalar on cell centers
    """
    plt.figure(figsize=(10, 5), dpi=140)

    extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]

    # robust symmetric color limits for ω
    clim = np.percentile(np.abs(omega), omega_percentile)
    clim = max(clim, 1e-12)

    # background ω
    plt.imshow(omega.T, origin="lower", extent=extent, aspect="auto",
               vmin=-clim, vmax=clim, cmap="RdBu_r")

    # overlay C contours (use original C, not gamma-corrected)
    # choose contour levels between min and max, but avoid a degenerate range
    cmin = float(np.min(C))
    cmax = float(np.max(C))
    if cmax - cmin < 1e-12:
        levels = np.linspace(0.0, 1.0, C_levels)
    else:
        levels = np.linspace(cmin + 0.05*(cmax - cmin), cmax - 0.05*(cmax - cmin), C_levels)

    plt.contour(x_centers, y_centers, C.T, levels=levels, linewidths=1.0)

    plt.xlabel("x")
    plt.ylabel("y")
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close()

def fig_to_rgb_array(fig):
    """
    Convert a Matplotlib figure to an RGB uint8 array (H, W, 3).
    Works with the Agg backend that is default in non-interactive scripts.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (H, W, 4)
    rgb = rgba[..., :3].copy()  # drop alpha
    return rgb

def render_frame_rgb(x_centers, y_centers, field, title, vmin=None, vmax=None,
                     ylim=None, cmap="RdBu_r"):
    """ 
    Render a field plot to an RGB uint8 array (H, W, 3).
    
    Parameters
    ----------
    field : (Nx, Ny) array, field on cell centers
    title : str, plot title
    vmin, vmax : float, optional, color limits
    ylim : tuple, optional, y-axis limits
    cmap : str, optional, colormap
    """
    fig = plt.figure(figsize=(10, 5), dpi=140)
    extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
    plt.imshow(field.T, origin="lower", extent=extent, aspect="auto",
               vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel("x")
    plt.ylabel("y")
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()

    frame = fig_to_rgb_array(fig)
    plt.close(fig)
    return frame

def render_overlay_rgb(x_centers, y_centers, omega, C, title,
                       omega_percentile=99.0, C_levels=10, ylim=None):
    """ 
    Render vorticity ω as background and overlay passive scalar C as contour lines
    to an RGB uint8 array (H, W, 3).
    
    Parameters
    ----------
    omega : (Nx, Ny) array, vorticity on cell centers
    C     : (Nx, Ny) array, passive scalar on cell centers
    title : str, plot title
    omega_percentile : float, percentile for robust color limits of ω
    C_levels : int, number of contour levels for C
    ylim : tuple, optional, y-axis limits
    """
    fig = plt.figure(figsize=(10, 5), dpi=140)
    extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]

    clim = np.percentile(np.abs(omega), omega_percentile)
    clim = max(clim, 1e-12)

    plt.imshow(omega.T, origin="lower", extent=extent, aspect="auto",
               vmin=-clim, vmax=clim, cmap="RdBu_r")

    cmin = float(np.min(C))
    cmax = float(np.max(C))
    if cmax - cmin < 1e-12:
        levels = np.linspace(0.0, 1.0, C_levels)
    else:
        levels = np.linspace(cmin + 0.05*(cmax - cmin), cmax - 0.05*(cmax - cmin), C_levels)

    plt.contour(x_centers, y_centers, C.T, levels=levels, linewidths=1.0)

    plt.xlabel("x")
    plt.ylabel("y")
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()

    frame = fig_to_rgb_array(fig)
    plt.close(fig)
    return frame
# %%  MAIN SIMULATION

# define output settings:
out_dir = "kh_fv_output"
frames_dir = os.path.join(out_dir, "frames")
frames_dir_C = os.path.join(out_dir, "frames_C")
frames_dir_overlay = os.path.join(out_dir, "frames_overlay")
frames_dir_zooms = os.path.join(out_dir, "frames_zooms")
frames_dir_C_zoom = os.path.join(out_dir, "frames_C_zoom")
frames_dir_overlay_zoom = os.path.join(out_dir, "frames_overlay_zoom")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(frames_dir_C, exist_ok=True)
os.makedirs(frames_dir_overlay, exist_ok=True)
os.makedirs(frames_dir_zooms, exist_ok=True)
os.makedirs(frames_dir_C_zoom, exist_ok=True)
os.makedirs(frames_dir_overlay_zoom, exist_ok=True)
gif_path = os.path.join(out_dir, "kh_finite_volume_omega.gif")
gif_path_C = os.path.join(out_dir, "kh_finite_volume_C.gif")
gif_path_overlay = os.path.join(out_dir, "kh_finite_volume_overlay.gif")

# simulation parameters:
Lx = 300.0 # domain size in x
Ly = 70.0  # domain size in y
Nx = 360*2   # grid points in x; start with 360; the higher the better but also computationally more expensive
Ny = 160*2   # grid points in y; start with 160; the higher the better but also computationally more expensive

dx = Lx / Nx # grid spacing in x
dy = Ly / Ny # grid spacing in y

x_centers = (np.arange(Nx) + 0.5) * dx
y_centers = (np.arange(Ny) + 0.5) * dy - 0.5 * Ly
y_faces_v = np.arange(Ny + 1) * dy - 0.5 * Ly

U0 = 1.5    # jet max velocity; start with 1.0
jet_w = 3.0  # jet half width
Re = 300.0  # start more viscous for stability, later increase again
nu = U0 * jet_w / Re  # kinematic viscosity
#kappa = 0.25 * nu  # passive scalar diffusivity
kappa = 0.00 * nu

eps = 0.06
f = 0.06
T_ramp = 8.0

CFL = 0.20
dt_max = 0.04

t_end = 800.0
target_frames = 220
frame_dt = t_end / target_frames

u, v, p, C = allocate_fields(Nx, Ny)

params = {
    "U0": U0,
    "jet_w": jet_w,
    "eps": eps,
    "f": f,
    "T_ramp": T_ramp,
    "y_centers": y_centers,
    "y_faces_v": y_faces_v,
}

# mild initial condition, almost rest, with a weak jet shape
u[:, :] = 0.0
v[:, :] = 0.0
C[:, :] = 0.0


# MAIN simulation/time-stepping loop:
t = 0.0
next_frame_t = 0.0
frames = []
frames_C = []
frames_overlay = []
frame_idx = 0

while t < t_end:
    # adaptive dt based on current max velocity
    umax = np.max(np.abs(u))
    vmax = np.max(np.abs(v))
    u_char = max(umax, vmax, 1e-6)
    dt_adv = CFL * min(dx, dy) / u_char
    dt_diff = 0.20 * min(dx, dy) ** 2 / max(nu, 1e-12)
    dt = min(dt_adv, dt_diff, dt_max, frame_dt)

    apply_bc(u, v, p, C, t=t, dt=dt, dx=dx, params=params)

    adv_u = advect_u(u, v, dx, dy)
    adv_v = advect_v(u, v, dx, dy)

    diff_u = nu * laplacian_u(u, dx, dy)
    diff_v = nu * laplacian_v(v, dx, dy)

    u_star = u + dt * (-adv_u + diff_u)
    v_star = v + dt * (-adv_v + diff_v)

    apply_bc(u_star, v_star, p, C, t=t, dt=dt, dx=dx, params=params)

    div_star = divergence(u_star, v_star, dx, dy)
    rhs = div_star / dt
    rhs -= np.mean(rhs)

    # diagnostic clamp to avoid catastrophic SOR RHS in case of reflections
    rhs_clip = 1e3
    rhs = np.clip(rhs, -rhs_clip, rhs_clip)
    
    # secure against NaNs/Infs:
    div_star = divergence(u_star, v_star, dx, dy)
    rhs = div_star / dt
    rhs -= np.mean(rhs)
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    # solve for pressure:
    p = solve_poisson_fft_y(p, rhs, dx, dy)

    u = u_star - dt * grad_p_to_u(p, dx)
    v = v_star - dt * grad_p_to_v(p, dy)

    apply_bc(u, v, p, C, t=t, dt=dt, dx=dx, params=params)

    adv_C = advect_scalar(C, u, v, dx, dy)
    diff_C = kappa * laplacian_center(C, dx, dy)
    C = C + dt * (-adv_C + diff_C)
    C = np.clip(C, 0.0, 1.0)
    apply_bc(u, v, p, C, t=t, dt=dt, dx=dx, params=params)

    if not (np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(p).all()):
        raise FloatingPointError("Non finite values. Further reduce CFL/dt_max or increase viscosity.")

    if t >= next_frame_t or t + dt >= t_end:
        omega_field = compute_vorticity(u, v, dx, dy)
        
        
        """ 
        we can plot two fields:
        1) omega_field: vorticity on cell centers
        2) C: passive scalar dye field
        Just secify which one to plot below via field_to_plot variable.
        """

        # omega plot:
        field_to_plot = omega_field
        title = f"2D KH, t = {t:7.2f}, Re = {Re:g}, dt = {dt:.3e} (omega field)"
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        frame_path_zoom = os.path.join(frames_dir_zooms, f"frame_{frame_idx:05d}.png")
        clim = np.percentile(np.abs(field_to_plot), 99.0)
        clim = max(clim, 1e-6)
        save_frame(frame_path, x_centers, y_centers, field_to_plot, title, vmin=-clim, vmax=clim)
        save_frame(frame_path_zoom, x_centers, y_centers, field_to_plot, title, vmin=-clim, vmax=clim,
                   ylim=(-15, 15))
        omega_rgb = render_frame_rgb(x_centers, y_centers, omega_field, title,
                             vmin=-clim, vmax=clim, cmap="RdBu_r")
        frames.append(omega_rgb)
        #frames.append(imageio.imread(frame_path))
        print(f"Saved {frame_path}")
        
        
        # plot C:
        field_to_plot = C
        # Gamma correction of c/field_to_pot for better visibility:
        field_to_plot_gamma = field_to_plot ** 0.6
        title = f"2D KH, t = {t:7.2f}, Re = {Re:g}, dt = {dt:.3e} (Passive Scalar C)"
        frame_path = os.path.join(frames_dir_C, f"frame_{frame_idx:05d}.png")
        frame_path_zoom = os.path.join(frames_dir_C_zoom, f"frame_{frame_idx:05d}.png")
        save_frame(frame_path, x_centers, y_centers, field_to_plot_gamma, title, vmin=0.0, vmax=1.0)
        save_frame(frame_path_zoom, x_centers, y_centers, field_to_plot_gamma, title, vmin=0.0, vmax=1.0,
                   ylim=(-15, 15))
        # render C to RAM:
        C_rgb = render_frame_rgb(x_centers, y_centers, field_to_plot_gamma, title,
                                vmin=0.0, vmax=1.0, cmap="Reds")
        frames_C.append(C_rgb)
        #frames_C.append(imageio.imread(frame_path))
        print(f"Saved {frame_path}")
        
        
        # overlay plot ω background + C contours:
        title = f"2D KH overlay, t = {t:7.2f}, Re = {Re:g}, dt = {dt:.3e} (ω + C contours)"
        frame_path = os.path.join(frames_dir_overlay, f"frame_{frame_idx:05d}.png")
        frame_path_zoom = os.path.join(frames_dir_overlay_zoom, f"frame_{frame_idx:05d}.png")
        save_overlay_frame(frame_path, x_centers, y_centers, omega_field, C, title, omega_percentile=99.0, C_levels=12)
        save_overlay_frame(frame_path_zoom, x_centers, y_centers, omega_field, C, title,
                           omega_percentile=99.0, C_levels=12, ylim=(-15, 15))
        # overlay: render to RAM
        overlay_rgb = render_overlay_rgb(x_centers, y_centers,
                                        omega=omega_field, C=C, title=title,
                                        omega_percentile=99.0, C_levels=12)
        frames_overlay.append(overlay_rgb)
        #frames_overlay.append(imageio.imread(frame_path))
        print(f"Saved {frame_path}")
        
        frame_idx += 1
        next_frame_t += frame_dt

    t += dt

imageio.mimsave(gif_path, frames, duration=0.06, plugin="pillow", palettesize=256, subrectangles=False)
imageio.mimsave(gif_path_C, frames_C, duration=0.06, plugin="pillow", palettesize=256, subrectangles=False)
imageio.mimsave(gif_path_overlay, frames_overlay, duration=0.06, plugin="pillow", palettesize=256, subrectangles=False)
print("All GIFs saved.")
# %% END
