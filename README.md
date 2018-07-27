# Laser
Python modules for laser simulation

25/07/2018

## List of modules:

- abcd:
  - Beam path calculation based on ABCD matrices, assuming paraxial approximation
- fn_gain
  - Amplifier gain simulation using Frantz-Nodvik formalism. Only titanium-sapphire cross-section data available for now
  - Need to add an example notebook
- fresnel_propag
  - Transverse 2D complex electric field propagator
  - Possibility to couple it to abcd.Beampath
- misc
  - Some general useful functions
- zernike
  - Module to generate and analyse laser wavefront based on the Zernike polynomials
  - Need to add an example notebook

## Installation

Download or clone this repository and navigate into the directroy Laser. There run 
`python setup.py install`.
