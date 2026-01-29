# %% IMPORTS
import os
import math
import numpy as np
import imageio.v2 as imageio

# Matplotlib nur zum Colormapping und zum PNG Speichern verwenden
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
# %% FUNCTIONS

def make_2d_wavenumbers(nx: int, ny: int, lx: float, ly: float):
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")
    k2 = kx**2 + ky**2
    return kx, ky, k2


def dealias_mask(nx: int, ny: int):
    """
    2/3 Regel in Fourier Raum, rechteckige Maske.
    """
    kx_cut = nx // 3
    ky_cut = ny // 3
    mask = np.ones((ny, nx), dtype=bool)
    # Indizes in FFT Ordnung: 0..N/2, -N/2+1..-1
    # Wir maskieren hohe Moden anhand der diskreten Mode Indizes.
    kx_idx = np.fft.fftfreq(nx) * nx
    ky_idx = np.fft.fftfreq(ny) * ny
    kx_i, ky_i = np.meshgrid(kx_idx, ky_idx, indexing="xy")
    mask &= (np.abs(kx_i) <= kx_cut)
    mask &= (np.abs(ky_i) <= ky_cut)
    return mask


def forcing_hat(nx: int, ny: int, kx, ky, k2, kf: float, width: float, amp: float, rng: np.random.Generator):
    """
    Bandbegrenzte, isotrope, stochastische Forcierung in einem Ring um |k| ~ kf.

    Wir forcieren die Vortizitätsgleichung direkt:
        d_t omega = ... + f
    Das ist numerisch robust und üblich für 2D Turbulenz.

    width ist die relative Bandbreite um kf.
    """
    k = np.sqrt(k2)
    band = (k > (1.0 - width) * kf) & (k < (1.0 + width) * kf)

    # Komplexes Rauschen, hermitesch so erzwingen, dass f im Realraum reell ist
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx))
    fhat = np.zeros((ny, nx), dtype=np.complex128)
    fhat[band] = np.exp(1j * phase[band])

    # Amplitude skalieren
    fhat *= amp

    # Hermitesche Symmetrie: fhat[-k] = conj(fhat[k])
    # Das erzwingen wir durch symmetrisches Mittel
    fhat = 0.5 * (fhat + np.conj(np.flip(np.flip(fhat, axis=0), axis=1)))

    # Nullmode nicht forcieren
    fhat[0, 0] = 0.0 + 0.0j
    return fhat


def rhs_vorticity(omega_hat, kx, ky, k2, nu: float, alpha: float, dealias: np.ndarray, f_hat):
    """
    Rechte Seite in Fourier Raum:
        d_t ω = - u·∇ω + ν ∇² ω - α ω + f

    u = (∂y ψ, -∂x ψ),  ∇² ψ = - ω
    """
    # Streamfunktion in Fourier Raum
    psi_hat = np.zeros_like(omega_hat)
    psi_hat[k2 != 0] = -omega_hat[k2 != 0] / k2[k2 != 0]
    psi_hat[0, 0] = 0.0 + 0.0j

    # Geschwindigkeit in Fourier Raum
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    # Zurück in Realraum
    u = np.fft.ifft2(u_hat).real
    v = np.fft.ifft2(v_hat).real
    omega = np.fft.ifft2(omega_hat).real

    # Gradienten von omega in Realraum über Fourier Ableitungen
    domega_dx = np.fft.ifft2(1j * kx * omega_hat).real
    domega_dy = np.fft.ifft2(1j * ky * omega_hat).real

    adv = u * domega_dx + v * domega_dy  # u·∇ω in Realraum

    # Fourier Transform der Advektion
    adv_hat = np.fft.fft2(adv)

    # Dealiasing auf nichtlinearer Term
    adv_hat *= dealias

    # Diffusion und lineare Reibung in Fourier Raum
    diff_hat = -nu * k2 * omega_hat
    drag_hat = -alpha * omega_hat

    return -adv_hat + diff_hat + drag_hat + f_hat


def omega_to_rgb(omega: np.ndarray, vlim: float, cmap_name: str = "RdBu_r"):
    """
    Omega Feld zu RGB uint8 mit symmetrischer Sättigung bei +/- vlim.
    """
    cmap = cm.get_cmap(cmap_name)
    x = np.clip(omega / vlim, -1.0, 1.0)
    x01 = 0.5 * (x + 1.0)
    rgba = cmap(x01)  # float in [0,1], shape (ny,nx,4)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def save_png(rgb: np.ndarray, path: str):
    """
    Speichere RGB als PNG ohne Achsen, ohne Rand.
    """
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def simulate_2d_cascade(
    nx: int = 256,
    ny: int = 256,
    lx: float = 2.0 * np.pi,
    ly: float = 2.0 * np.pi,
    nu: float = 2e-4,
    alpha: float = 0.0,
    dt: float = 2.5e-3,
    nsteps: int = 6000,
    frame_every: int = 10,
    seed: int = 0,
    forcing_kf: float = 10.0,
    forcing_width: float = 0.15,
    forcing_amp: float = 5e2,
    out_dir: str = "frames_2d_cascade",
    gif_path: str = "cascade_2d.gif",
    gif_fps: int = 30,
    vlim: float = 8.0):
    """
    Führt eine 2D Turbulenz Simulation aus und erzeugt Frames.

    Parameter Hinweise:
    * nu kleiner -> stärkere kleinräumige Strukturen, aber dt muss kleiner sein.
    * forcing_kf bestimmt die Einspeiseskala. Größer -> Forcierung auf kleineren Skalen.
    * alpha > 0 hilft, dass die größten Skalen nicht unphysikalisch Energie ansammeln (inverse Kaskade).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    kx, ky, k2 = make_2d_wavenumbers(nx, ny, lx, ly)
    dealias = dealias_mask(nx, ny)

    # Anfangsbedingung: kleines Rauschen in omega
    omega0 = 0.1 * rng.standard_normal((ny, nx))
    omega_hat = np.fft.fft2(omega0)

    frames_rgb = []

    nframes = 0
    for step in range(nsteps + 1):
        # step = 0
        # Forcierung pro Schritt neu ziehen (stochastisch)
        f_hat = forcing_hat(nx, ny, kx, ky, k2, kf=forcing_kf, width=forcing_width, amp=forcing_amp, rng=rng)

        # RK2
        k1 = rhs_vorticity(omega_hat, kx, ky, k2, nu=nu, alpha=alpha, dealias=dealias, f_hat=f_hat)
        omega_hat_mid = omega_hat + 0.5 * dt * k1
        k2_rhs = rhs_vorticity(omega_hat_mid, kx, ky, k2, nu=nu, alpha=alpha, dealias=dealias, f_hat=f_hat)
        omega_hat = omega_hat + dt * k2_rhs

        # Optional: Dealiasing auch auf omega selbst (robuster)
        omega_hat *= dealias

        if step % frame_every == 0:
            omega = np.fft.ifft2(omega_hat).real
            rgb = omega_to_rgb(omega, vlim=vlim, cmap_name="RdBu_r")

            # In RAM halten
            frames_rgb.append(rgb)

            # Auf Platte speichern
            png_path = os.path.join(out_dir, f"frame_{nframes:06d}.png")
            save_png(rgb, png_path)
            #save_png(omega, png_path)

            if (nframes % 50) == 0:
                print(f"Saved frame {nframes} at step {step}/{nsteps}")

            nframes += 1

    # GIF erstellen
    imageio.mimsave(gif_path, frames_rgb, fps=gif_fps)
    print(f"Done. Wrote {nframes} frames, GIF saved to: {gif_path}")
    return frames_rgb

# %% MAIN SIMULATION

simulate_2d_cascade(
    nx=256,
    ny=256,
    nu=2e-4,
    alpha=1e-2,          # hilft bei inverse Kaskade Sättigung
    dt=2.5e-3,
    nsteps=6000,
    frame_every=10,
    forcing_kf=10.0,
    forcing_width=0.15,
    forcing_amp=5e2,
    out_dir="frames_2d_cascade",
    gif_path="cascade_2d.gif",
    gif_fps=30,
    vlim=8.0,
    seed=1,
)

""" DEBUG:
    nx=256
    ny=256
    nu=2e-4
    alpha=1e-2
    dt=2.5e-3
    nsteps=6000,
    frame_every=10
    forcing_kf=10.0
    forcing_width=0.15
    forcing_amp=5e2
    out_dir="frames_2d_cascade"
    gif_path="cascade_2d.gif"
    gif_fps=30
    vlim=8.0
    seed=1
    lx = 2.0 * np.pi
    ly = 2.0 * np.pi
"""