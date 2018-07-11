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

def cart2pol(x,y):
    """Convert cartesian to polar coordinates"""
    return np.abs(x+1j*y), np.angle(x+1j*y)

def pol2cart(r,theta):
    """Convert polar to cartesian coodinates"""
    return np.real(r*exp(1j*theta)), np.imag(r*exp(1j*theta))

def array_trim(ar):
    """Trim zeros of 2D map"""
    ar_trim = ar.copy()
    ar_trim = ar_trim[:, ar_trim.any(axis=0)] # trim columns
    ar_trim = ar_trim[ar_trim.any(axis=1), :] # trim rows
    return ar_trim