""" 
Script to create a schematic Richardson cascade diagram.

author: Fabrizio Musacchio
date: Feb 2021 / Jan 2026
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrow
# %% FUNCTIONS
def blob_path(center=(0.0, 0.0), r=1.0, n=200, jitter=0.25, seed=None):
    """
    Create an irregular closed curve ("blob") as a Matplotlib Path.

    Parameters
    ----------
    center : tuple of float
        Blob center (x, y).
    r : float
        Mean radius.
    n : int
        Number of support points along the contour.
    jitter : float
        Relative amplitude variation of the radius.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    Path
        Closed path describing the blob outline.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Smooth random radius function via a small number of Fourier modes
    m = 6
    a = rng.normal(0, 1, m)
    b = rng.normal(0, 1, m)

    radial = np.ones_like(theta)
    for k in range(1, m + 1):
        radial += (jitter / (k**1.2)) * (a[k - 1] * np.cos(k * theta) + b[k - 1] * np.sin(k * theta))

    radial = np.clip(radial, 0.4, None)

    x = center[0] + r * radial * np.cos(theta)
    y = center[1] + r * radial * np.sin(theta)

    verts = np.column_stack([x, y])
    verts = np.vstack([verts, verts[0]])  # close contour

    codes = np.full(len(verts), Path.LINETO, dtype=int)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    return Path(verts, codes)


def draw_blob(ax, center, r, lw=4, color="black", seed=None):
    """
    Draw a blob outline on the given axes.
    """
    p = blob_path(center=center, r=r, jitter=0.28, seed=seed)
    patch = PathPatch(
        p,
        facecolor="none",
        edgecolor=color,
        lw=lw,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)


def draw_dissipation_dots(ax, y, x0, x1, n=38, size=8):
    """
    Draw a row of small squares suggesting the dissipation range.
    """
    xs = np.linspace(x0, x1, n)
    ax.scatter(xs, np.full_like(xs, y), s=size, c="black", marker="s")


def place_nonoverlapping_centers(
    rng,
    n,
    y,
    x_min,
    x_max,
    r0,
    dr,
    pad=2.2,
    y_jitter=0.05,
    max_tries=20000,
):
    """
    Place n blob centers (x, y, r) with simple non-overlap constraints.

    We approximate each blob as a circle of radius r for collision checking.
    A candidate is accepted only if its center is sufficiently far from all
    existing centers in that row.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n : int
        Target number of blobs to place.
    y : float
        Nominal row y coordinate.
    x_min, x_max : float
        Allowed x range for placing centers.
    r0 : float
        Mean radius of blobs in this row.
    dr : float
        Radius scatter (Gaussian).
    pad : float
        Minimum center-to-center distance factor relative to (r_i + r_j).
        Larger values create more spacing.
    y_jitter : float
        Vertical jitter (Gaussian) applied to each placed center.
    max_tries : int
        Maximum number of placement attempts.

    Returns
    -------
    list of tuples
        List of (x, y, r) tuples.
    """
    centers = []
    tries = 0

    while (len(centers) < n) and (tries < max_tries):
        tries += 1

        r = max(0.06, r0 + rng.normal(0, dr))
        x = rng.uniform(x_min, x_max)
        yy = y + rng.normal(0, y_jitter)

        ok = True
        for (xj, yj, rj) in centers:
            d2 = (x - xj) ** 2 + (yy - yj) ** 2
            dmin = pad * (r + rj)
            if d2 < dmin**2:
                ok = False
                break

        if ok:
            centers.append((x, yy, r))

    return centers


def add_swirl_arrow(ax, center, r, rng, color="black", lw=2.2):
    """
    Add a curved arrow inside a blob to indicate rotation direction.

    This is a purely schematic visual cue (clockwise vs counterclockwise).
    """
    x, y = center

    # Choose rotation sense: -1 (clockwise), +1 (counterclockwise)
    sgn = rng.choice([-1.0, 1.0])

    # Choose an arc segment on an inner circle
    a0 = rng.uniform(0.0, 2.0 * np.pi)
    da = sgn * rng.uniform(np.pi / 2.5, np.pi / 1.7)

    r_in = 0.55 * r
    x0, y0 = x + r_in * np.cos(a0), y + r_in * np.sin(a0)
    x1, y1 = x + r_in * np.cos(a0 + da), y + r_in * np.sin(a0 + da)

    # Curvature sign controls arc direction
    rad = 0.35 * sgn

    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def make_richardson_cartoon(rows=None, seed=1, savepath=None,
                            figsize=(10, 6)):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    
    if rows is None:
        # Row parameters: (y, count, mean radius, radius std)
        rows = [
            (4.8, 3, 0.70, 0.15),    # large eddies
            (3.45, 7, 0.32, 0.07),   # medium
            (2.55, 10, 0.22, 0.05),  # smaller
            (1.85, 14, 0.14, 0.03),  # even smaller
        ]

    # Place blobs row-by-row with non-overlap constraints
    for y, n, r0, dr in rows:
        centers = place_nonoverlapping_centers(
            rng=rng,
            n=n,
            y=y,
            x_min=1.2,
            x_max=8.8,
            r0=r0,
            dr=dr,
            pad=2.2,
            y_jitter=0.06,
            max_tries=30000)

        for (x, yy, r) in centers:
            draw_blob(ax, (x, yy), r, lw=4, seed=rng.integers(1_000_000))

            # Add rotation arrows only for sufficiently large blobs, and not for all of them
            # if (r > 0.18) and (rng.random() < 0.65):
            #     add_swirl_arrow(ax, (x, yy), r, rng=rng, lw=2.2)

    # Dissipation dots (bottom "pixel row")
    #draw_dissipation_dots(ax, y=0.85, x0=0.9, x1=9.1, n=55, size=5)

    # Dotted vertical marker in the middle
    ax.plot([5.2, 5.2], [0.85, 1.45], color="black", lw=2.5, linestyle=":")

    # green energy arrow on the left:
    arrow = FancyArrow(
        0.6, 5.3, 0.0, -4.5,
        width=0.08,
        head_width=0.35,
        head_length=0.35,
        length_includes_head=True,
        color="green")
    ax.add_patch(arrow)
    ax.text(0.15, 3.3, "E", color="green", fontsize=20, fontweight="bold")

    # labels:
    ax.text(1.1, 5.85, "Energy input range", color="red", fontsize=14)
    ax.text(1.05, 2.9, "Inertial range", color="black", fontsize=13)
    ax.text(1.3, 0.15, "Energy dissipation range", color="red", fontsize=14)

    # red arrows (top and bottom):
    ax.annotate(
        "",
        xy=(1.4, 5.45),
        xytext=(0.8, 5.95),
        arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
    ax.annotate(
        "",
        xy=(0.8, 0.00),
        xytext=(1.4, 0.50),
        arrowprops=dict(arrowstyle="->", color="red", lw=2.5))

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


# %% MAIN
# Row parameters: (y, count, mean radius, radius std)
rows = [
    (5.1, 5, 0.70, 0.05),    # large eddies
    (3.45, 7, 0.32, 0.02),   # medium
    (2.55, 10, 0.22, 0.02),  # smaller
    (1.85, 14, 0.12, 0.01),  # even smaller
    (0.75, 25, 0.05, 0.001),  # dissipation range
]
make_richardson_cartoon(rows=rows,
                        seed=39,
                        figsize=(7, 6),
                        savepath="richardson_cascade_scheme.png")
# %%
