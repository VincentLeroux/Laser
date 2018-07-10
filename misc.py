import numpy as np

def gauss2D(x, y, fwhmx, fwhmy, x0=0, y0 = 0, offset=0, order=1, int_FWHM = True):
    """
    Define a (super-)Gaussian 2D beam
    """
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    return np.exp(-np.log(2)*coeff*((2*(x-x0)/fwhmx)**2 + (2*(y-y0)/fwhmy)**2)**order) + offset

def gauss1D(x, fwhm, x0=0, offset=0, order=1, int_FWHM = True):
    """
    Define a (super-)Gaussian 1D beam
    """
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    return np.exp(-np.log(2)*coeff*(2*(x-x0)/fwhm)**(2*order)) + offset