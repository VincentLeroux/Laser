# laser
Python modules for laser simulation, contained in the folder __laser__

14/08/2018

## List of modules:

- __abcd__
  - Beam path calculation based on ABCD matrices, assuming paraxial approximation
- __fn_gain__
  - Amplifier gain simulation using Frantz-Nodvik formalism. Only titanium-sapphire cross-section data available for now
  - Need to add an example notebook
- __fresnel_propag__
  - Transverse 2D complex electric field propagator
  - Possibility to couple it to abcd.Beampath
- __misc__
  - Some general useful functions
- __plot_utils__
  - Some useful functions to plot data
- __zernike__
  - Module to generate and analyse laser wavefront based on the Zernike polynomials

## Installation

Download or clone this repository and navigate into the directroy laser. There run 
`python setup.py install`.
