"""
This code simulates the von Kármán vortex street behind a cylinder using the
Lattice-Boltzmann Method (LBM) in Python with NumPy. The core is based on the 
original implementation by Felix Köhler, which was based on JAX:

https://github.com/Ceyron/machine-learning-and-simulation

I simply translated the JAX code to NumPy for educational purposes. All
credits go to Felix Köhler whos shared his code under MIT license.

author: Felix Köhler (original JAX code)
        Fabrizio Musacchio (NumPy translation)
date: Feb 2021 / Jan 2026
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
# %% PARAMETERS
N_ITERATIONS = 25_000
REYNOLDS_NUMBER = 10

N_POINTS_X = 300
N_POINTS_Y = 50

CYLINDER_CENTER_INDEX_X = N_POINTS_X // 5
CYLINDER_CENTER_INDEX_Y = N_POINTS_Y // 2
CYLINDER_RADIUS_INDICES = N_POINTS_Y // 9

MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04

VISUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITERATIONS = 0000

MAKE_GIF = True
GIF_PATH = f"von_karman_numpy_Re{REYNOLDS_NUMBER}.gif"
GIF_FPS = 20

RANDOM_SEED = 1

# create output folder for frames:
FRAMES_PATH = f"frames_Re{REYNOLDS_NUMBER}"
os.makedirs(FRAMES_PATH, exist_ok=True)
# %% LATTICE SETUP
N_DISCRETE_VELOCITIES = 9

# velocities as (cx, cy) per direction i:
LATTICE_VELOCITIES = np.array(
    [
        [0, 0],   # 0
        [1, 0],   # 1
        [0, 1],   # 2
        [-1, 0],  # 3
        [0, -1],  # 4
        [1, 1],   # 5
        [-1, 1],  # 6
        [-1, -1], # 7
        [1, -1],  # 8
    ],
    dtype=int
)

OPPOSITE_LATTICE_INDICES = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

LATTICE_WEIGHTS = np.array(
    [
        4/9,
        1/9, 1/9, 1/9, 1/9,
        1/36, 1/36, 1/36, 1/36,
    ],
    dtype=float
)

RIGHT_VELOCITIES = np.array([1, 5, 8], dtype=int)
UP_VELOCITIES = np.array([2, 5, 6], dtype=int)
LEFT_VELOCITIES = np.array([3, 6, 7], dtype=int)
DOWN_VELOCITIES = np.array([4, 7, 8], dtype=int)

PURE_VERTICAL_VELOCITIES = np.array([0, 2, 4], dtype=int)
PURE_HORIZONTAL_VELOCITIES = np.array([0, 1, 3], dtype=int)
# %% HELPER FUNCTIONS
def make_cylinder_mask(nx: int, ny: int, cx: int, cy: int, r: int) -> np.ndarray:
    """ 
    Create a boolean mask for a cylinder in a 2D grid.
    """
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2

def get_density(f: np.ndarray) -> np.ndarray:
    """ 
    Compute macroscopic density from distribution functions.
    """
    return np.sum(f, axis=-1)

def get_macroscopic_velocities(f: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ 
    Compute macroscopic velocities from distribution functions and density.
    """
    jx = np.sum(f * LATTICE_VELOCITIES[None, None, :, 0], axis=-1)
    jy = np.sum(f * LATTICE_VELOCITIES[None, None, :, 1], axis=-1)
    u = np.zeros((f.shape[0], f.shape[1], 2), dtype=float)
    u[..., 0] = jx / rho
    u[..., 1] = jy / rho
    return u

def get_equilibrium_discrete_velocities(u: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """ 
    Compute equilibrium distribution functions for given macroscopic velocities and density.
    """
    ux = u[..., 0]  # (nx, ny)
    uy = u[..., 1]  # (nx, ny)

    # cu = c_i · u:
    cu = (
        LATTICE_VELOCITIES[None, None, :, 0] * ux[..., None]
        +
        LATTICE_VELOCITIES[None, None, :, 1] * uy[..., None]
    )  # (nx, ny, 9)

    uu = ux**2 + uy**2  # (nx, ny)

    feq = (
        rho[..., None]
        * LATTICE_WEIGHTS[None, None, :]
        * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * uu[..., None]))
    
    return feq

def compute_vorticity(u: np.ndarray) -> np.ndarray:
    """ 
    Compute the z-component of the vorticity (curl) of a 2D velocity field.
    
    u shape: (nx, ny, 2). We compute curl_z = d u_y / d x - d u_x / d y
    """
    dux_dx, dux_dy = np.gradient(u[..., 0], axis=(0, 1))
    duy_dx, duy_dy = np.gradient(u[..., 1], axis=(0, 1))
    return (duy_dx - dux_dy)
# %% MAIN RUN

# setup random number generator:
rng = np.random.default_rng(RANDOM_SEED)

# this follows the logic of the JAX code:
# nu = (U_max * R) / Re
# omega = 1 / (3 nu + 0.5)
kinematic_viscosity = (MAX_HORIZONTAL_INFLOW_VELOCITY * CYLINDER_RADIUS_INDICES) / REYNOLDS_NUMBER
relaxation_omega = 1.0 / (3.0 * kinematic_viscosity + 0.5)

# domain mask:
obstacle_mask = make_cylinder_mask(
    N_POINTS_X, N_POINTS_Y,
    CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y,
    CYLINDER_RADIUS_INDICES)


""" We add explicit no-slip walls top and bottom:
The JAX code implicitly treats y=0 and y=ny-1 as special by not prescribing inflow there.
Making them solid stabilizes the channel and prevents vertical wrap-around artifacts. 
"""
wall_mask = np.zeros((N_POINTS_X, N_POINTS_Y), dtype=bool)
wall_mask[:, 0] = True
wall_mask[:, -1] = True

solid_mask = obstacle_mask | wall_mask

# inflow profile u(x=0, y, :) prescribed for y=1:-1:
velocity_profile = np.zeros((N_POINTS_X, N_POINTS_Y, 2), dtype=float)
velocity_profile[:, :, 0] = MAX_HORIZONTAL_INFLOW_VELOCITY
velocity_profile[:, :, 1] = 0.0

# initialize distributions to equilibrium of uniform inflow:
rho0 = np.ones((N_POINTS_X, N_POINTS_Y), dtype=float)
u0 = velocity_profile.copy()
u0[solid_mask, :] = 0.0
f = get_equilibrium_discrete_velocities(u0, rho0)

# small perturbation near cylinder to break symmetry:
x0 = CYLINDER_CENTER_INDEX_X + CYLINDER_RADIUS_INDICES + 2
x1 = min(N_POINTS_X, x0 + 30)
y0 = max(1, CYLINDER_CENTER_INDEX_Y - 10)
y1 = min(N_POINTS_Y - 1, CYLINDER_CENTER_INDEX_Y + 10)
u0[x0:x1, y0:y1, 1] += 1e-4 * (rng.random((x1 - x0, y1 - y0)) - 0.5)
f = get_equilibrium_discrete_velocities(u0, rho0)

frames_rgb = []

for iteration_index in tqdm(range(N_ITERATIONS)):
    f_prev = f

    # (1) outflow BC at right boundary: copy left-moving populations from x = -2 to x = -1:
    f_prev[-1, :, LEFT_VELOCITIES] = f_prev[-2, :, LEFT_VELOCITIES]

    # (2) Macroscopic fields
    rho = get_density(f_prev)
    u = get_macroscopic_velocities(f_prev, rho)

    # enforce no-slip in macroscopic fields on solids (helps diagnostics and stability):
    u[solid_mask, :] = 0.0

    # (3) Zou/He velocity inlet at x=0 for y=1:-1:
    u[0, 1:-1, :] = velocity_profile[0, 1:-1, :]

    # reconstruct inlet density (same formula as in the JAX code):
    # rho_in = (sum f_i over pure vertical + 2 sum f_i over left movers) / (1 - u_x)
    f0 = f_prev[0, :, :]                  # shape should be (ny, 9)
    sum_pure_vertical = f0[:, PURE_VERTICAL_VELOCITIES].sum(axis=1)  # (ny,)
    sum_left = f0[:, LEFT_VELOCITIES].sum(axis=1)                    # (ny,)
    rho_in = (sum_pure_vertical + 2.0 * sum_left) / (1.0 - u[0, :, 0])
    rho[0, :] = rho_in

    # walls at inlet should remain no-slip:
    u[0, 0, :] = 0.0
    u[0, -1, :] = 0.0

    # (4) equilibrium:
    feq = get_equilibrium_discrete_velocities(u, rho)

    # (3b) Zou/He completion: set right-moving populations at inlet from equilibrium
    f_prev[0, :, RIGHT_VELOCITIES] = feq[0, :, RIGHT_VELOCITIES]

    # (5) BGK collision:
    f_post = f_prev - relaxation_omega * (f_prev - feq)

    # (6) bounce-back on solids, using pre-collision populations as in the JAX version:
    for i in range(N_DISCRETE_VELOCITIES):
        f_post[solid_mask, i] = f_prev[solid_mask, OPPOSITE_LATTICE_INDICES[i]]

    # (7) streaming (periodic roll in x and y):
    # walls block physical wrap-around through bounce-back, but the roll is convenient and standard.
    f_stream = f_post.copy()
    for i in range(N_DISCRETE_VELOCITIES):
        cx, cy = LATTICE_VELOCITIES[i]
        f_stream[:, :, i] = np.roll(np.roll(f_post[:, :, i], shift=cx, axis=0), shift=cy, axis=1)

    f = f_stream

    # diagnostics for blow-up:
    if iteration_index % 200 == 0:
        rho_chk = get_density(f)
        if not np.isfinite(rho_chk).all():
            print(f"Non-finite density at iteration {iteration_index}. Aborting.")
            break

    # plotting:
    if VISUALIZE and iteration_index % PLOT_EVERY_N_STEPS == 0 and iteration_index > SKIP_FIRST_N_ITERATIONS:
        rho_vis = get_density(f)
        u_vis = get_macroscopic_velocities(f, rho_vis)
        u_vis[solid_mask, :] = 0.0

        speed = np.linalg.norm(u_vis, axis=-1)
        vort = compute_vorticity(u_vis)

        fig = plt.figure(figsize=(15, 6))

        ax1 = plt.subplot(211)
        im1 = ax1.contourf(
            speed.T,
            levels=50,
            cmap="viridis")
        plt.colorbar(im1, ax=ax1).set_label("velocity magnitude")
        ax1.add_patch(
            plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES,
                color="white",
                zorder=5))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"speed, iter={iteration_index}, Re≈{REYNOLDS_NUMBER}")

        ax2 = plt.subplot(212)
        """ im2 = ax2.contourf(
            vort.T,
            levels=50,
            cmap="RdBu_r",
            vmin=-0.02,
            vmax=0.02,
        )
        plt.colorbar(im2, ax=ax2).set_label("vorticity (curl z)") """
        im2 = ax2.imshow(
            vort.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-0.02,
            vmax=0.02,
            interpolation="bilinear",
            aspect="auto")
        plt.colorbar(im2, ax=ax2).set_label("vorticity (curl z)")
        ax2.add_patch(
            plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES,
                color="white",
                zorder=5))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("vorticity")

        plt.tight_layout()

        # render figure to RGB array (in-memory):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        rgb = buf[:, :, :3].copy()   # drop alpha channel

        if MAKE_GIF:
            frames_rgb.append(rgb)

        # also save current frame as PNGs for inspection:
        frame_path = os.path.join(FRAMES_PATH, f"frame_{iteration_index:05d}.png")
        plt.savefig(frame_path, dpi=300)

        plt.close(fig)

if MAKE_GIF and len(frames_rgb) > 0:
    imageio.mimsave(GIF_PATH, frames_rgb, fps=GIF_FPS)
    print(f"Saved GIF to: {GIF_PATH}")
# %% END