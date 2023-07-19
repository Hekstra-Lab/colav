<!---
.. colav documentation master file, created by
   sphinx-quickstart on Thu Jul 13 21:49:07 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
-->

# Welcome to `colav`'s documentation!

Welcome to `colav`, an open-source Python package that implements three feature extraction methods (dihedral angles, C\alpha pairwise distances, and strain analysis) for representing protein structures in the PDB format. This package was primarily built to analyze medium to large (>100 & <10,000) datasets composed of individual protein structures. In particular, `colav` may be useful to analyze datasets from a crystallographic drug fragment screen (see our forthcoming paper!). 

The [quickstart](quickstart.md) provides installation instructions and introduces the feature extraction methods in more detail with examples using protein tyrosine phosphatase 1B (PTP-1B). 

We are adding command-line options for ease of use! Check back soon for more information. 

## Table of Contents
```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:
      
   extract_data.rst
   internal_coordinates.rst
   strain_analysis.rst
```

## Indices
```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```