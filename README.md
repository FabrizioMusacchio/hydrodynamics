# Hydrodynamics: A collection of educational Python scripts

![GIF](kelvin_helmholtz_instability/kh_fv_output/kh_finite_volume_C_loop.gif)


This repository contains Python scripts for different examples from the field of hydrodynamics. Each scripts belongs to one of the following blog posts, which provide detailed explanations of the implemented physics and numerical methods:


* [Hydrodynamics: A brief overview of fluid dynamics and its fundamental equations](https://www.fabriziomusacchio.com/blog/2021-03-04-hydrodynamics/) (overview article)
* [Kelvin–Helmholtz instability in 2D incompressible shear flows](https://www.fabriziomusacchio.com/blog/2021-03-02-kelvin_helmholtz_instability/)
  * ⟶ `kelvin_helmholtz_instability/`
* [A spatially developing 2D Kelvin Helmholtz jet with a finite volume projection method](https://www.fabriziomusacchio.com/blog/2021-03-03-kelvin_helmholtz_instability_via_finite_volume/)
  * ⟶ `kelvin_helmholtz_instability/`
* [The von Kármán vortex street](https://www.fabriziomusacchio.com/blog/2021-03-10-karman_vortex_street/)
  * ⟶ `karman_vortex_street/`
* [Turbulence, Richardson cascade, and spectral scaling in incompressible flows](https://www.fabriziomusacchio.com/blog/2021-03-06-turbulence/)
  * ⟶ `richardson_cascade/`
* [Forced 2D turbulence and Richardson cascade in a pseudospectral vorticity solver](https://www.fabriziomusacchio.com/blog/2021-03-07-richardson_cascade/)
  * ⟶ `richardson_cascade/`
* [Wavelet analysis in turbulence (and beyond)](https://www.fabriziomusacchio.com/blog/2021-03-18-wavelet_analysis_in_turbulence/)
  * ⟶ `wavelet_analysis/`

The scripts are intended as didactic and conceptual examples. They prioritize clarity and physical transparency over numerical efficiency or large-scale applicability. The focus is on illustrating fundamental mechanisms of hydrodynamics and standard modeling approaches rather than providing optimized or fully general simulation frameworks.

Many scripts deliberately rely on reduced models, idealized geometries, or simplified boundary conditions. These choices are made to keep the connection between equations, numerical implementation, and physical interpretation as direct as possible.

The repository reflects the state of the accompanying blog series and may evolve over time. Backward compatibility is not guaranteed, but changes are typically driven by conceptual clarification rather than feature expansion.

![GIF](richardson_cascade/cascade_2d_spectrum.gif_kf24.0_1024_looped.gif)

## Installation
For reproducibility, create a new conda environment with the following packages:

```bash
conda create -n hydrodynamics python=3.12 mamba -y
conda activate hydrodynamics
mamba install -y numpy matplotlib scipy imageio ffmpeg pywavelets ipykernel ipython
```

## Usage
Each script can be run directly using the Python environment described above. In particular, they are written in such a way, that they can be interactively executed cell-by-cell, e.g., in VS Code's interactive window. You can also place them in a Jupyter notebook for step-by-step execution.

## Citation
If you use code from this repository for your own research, teaching material, or derived software, please consider citing the Zenodo archive associated with this repository. Proper citation helps acknowledge the original source, provides context for the implemented physical models and numerical assumptions, and supports reproducibility.

When appropriate, citing the specific blog post that discusses the underlying physics and numerical methods in detail is encouraged in addition to the repository itself.

If you use substantial parts of the code in an academic publication, a reference to both the repository and the associated blog article is recommended.

Here is the suggested citation format for the repository:

> Musacchio, F. (2026). *Hydrodynamics: A collection of educational Python scripts*. Zenodo. https://doi.org/10.5281/zenodo.18411283

```bibtex
@software{musacchio_hydrodynamics_2026,
  author       = {Musacchio, Fabrizio},
  title        = {Hydrodynamics: A collection of educational Python scripts},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18411283},
  url          = {https://doi.org/10.5281/zenodo.18411283}
}
```


Thank you for considering proper citation practices.

## Contact and support
For questions or suggestions, please open an issue on GitHub or contact the author via email: [Fabrizio Musacchio](mailto:fabrizio.musacchio@posteo.de)


