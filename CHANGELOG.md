# Release notes for the Hydrodynamics repository

## 🚀 Release v1.0.0
This release marks the first complete and stable publication of the *Hydrodynamics* educational script collection. It consolidates all Python scripts used throughout the accompanying [hydrodynamics blog series](https://www.fabriziomusacchio.com/blog/2021-03-04-hydrodynamics/) into a single, citable repository.

Version v1.0.0 establishes a coherent reference point for teaching, reuse, and reproducible exploration of fundamental concepts in hydrodynamics.

### 📦 Scope and content
This release includes educational Python scripts covering a broad range of core topics in hydrodynamics, from Kelvin-Helmholtz instabilities to turbulence and wavelet analysis. Each script is directly associated with a dedicated blog post that provides detailed physical and mathematical context.

Included topics are:

* Kelvin-Helmholtz instability in shear flows
* Von Kármán vortex streets behind obstacles
* Richardson cascade and spectral scaling in 2D turbulence
* Wavelet analysis techniques applied to turbulent flows

The repository structure reflects this thematic organization and mirrors the progression of the blog series.

### 🧠 Conceptual focus
The scripts in this repository are designed as **didactic and conceptual examples**. Emphasis is placed on:

* physical transparency
* direct correspondence between equations and code
* minimal numerical and algorithmic overhead
* clarity over computational performance

Many models deliberately rely on reduced geometries, simplified boundary conditions, or idealized assumptions to keep the physical mechanisms explicit.

### 🔬 Reproducibility and usage
All scripts are compatible with a lightweight Python environment based on NumPy, SciPy, PyWavelets, and Matplotlib. They are written to support both direct execution and interactive, cell-by-cell exploration in development environments such as VS Code or Jupyter.

This release provides a stable baseline for reuse in:

* teaching and coursework
* self-study
* illustrative figures and animations
* methodological extensions

Backward compatibility across future releases is not guaranteed, but changes will primarily serve conceptual clarification rather than feature expansion.

### 📖 Citation and archiving
This release is archived on [Zenodo](https://doi.org/10.5281/zenodo.18411283) and assigned a DOI (10.5281/zenodo.18411283), making it citable in academic contexts.

Users of this repository are encouraged to cite the Zenodo record and, where appropriate, the corresponding blog posts that document the physical background and numerical choices in detail. How to cite is provided in the README and in the CITATION.cff file.

### 🔖 Versioning note
Version v1.0.0 supersedes the initial v0.0.1 placeholder release, which primarily established the repository structure. No prior code is deprecated by this release.

### 📝 License
All code is released under the GPL-3.0 License.

### ✨ Outlook
Future releases may expand individual examples, refine numerical implementations, or add complementary scripts aligned with new blog posts. Any such extensions will build on the conceptual baseline established with this release.