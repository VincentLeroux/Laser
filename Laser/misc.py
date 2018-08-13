import numpy as np


def gauss2D(x, y, fwhmx, fwhmy, x0=0, y0=0, offset=0, order=1, int_FWHM=True):
    """
    Define a (super-)Gaussian 2D beam. Identical to laser.misc.gauss2D.

    Parameters
    ----------
    x: float 2D np.array
        Horizontal axis of the Gaussian

    y: float 2D np.array
        Vertical axis of the Gaussian

    fwhmx: float
        Horizontal Full Width at Half Maximum

    fwhmy: float
        Vertical Full Width at Half Maximum

    x0: float, optional
        Horizontal center position of the Gaussian

    y0: float, optional
        Vertical center position of the Gaussian

    offset: float, optional
        Amplitude offset of the Gaussian

    order: int, optional
        order of the super-Gaussian function.
        Defined as: exp( - ( x**2 + y**2 )**order )

    int_FWHM: boolean, optional
        If True, the FWHM is the FWHM of the square of the Gaussian (intensity).
        If False, it is the FWHM of the Gaussian directly (electric field).
    """
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    return np.exp(-np.log(2) * coeff * ((2 * (x - x0) / fwhmx)**2 + (2 * (y - y0) / fwhmy)**2)**order) + offset


def gauss1D(x, fwhm, x0=0, offset=0, order=1, int_FWHM=True):
    """
    Define a (super-)Gaussian 1D beam. Identical to laser.misc.gauss2D.

    Parameters
    ----------
    x: float 1D np.array
        Axis of the Gaussian

    fwhm: float
        Full Width at Half Maximum

    x0: float, optional
        Center position of the Gaussian

    offset: float, optional
        Amplitude offset of the Gaussian

    order: int, optional
        order of the super-Gaussian function.
        Defined as: exp( - ( x**2 )**order )

    int_FWHM: boolean, optional
        If True, the FWHM is the FWHM of the square of the Gaussian (intensity).
        If False, it is the FWHM of the Gaussian directly (electric field).
    """
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    return np.exp(-np.log(2) * coeff * ((2 * (x - x0) / fwhm)**2)**order) + offset


def cart2pol(x, y):
    """Convert cartesian to polar coordinates"""
    return np.abs(x + 1j * y), np.angle(x + 1j * y)


def pol2cart(r, theta):
    """Convert polar to cartesian coodinates"""
    return np.real(r * exp(1j * theta)), np.imag(r * exp(1j * theta))


def array_trim(ar):
    """Trim zeros of 2D map"""
    ar_trim = ar.copy()
    ar_trim = ar_trim[:, ar_trim.any(axis=0)]  # trim columns
    ar_trim = ar_trim[ar_trim.any(axis=1), :]  # trim rows
    return ar_trim


def vect(N):
    """Returns a centered array between -0.5 and 0.5"""
    return np.linspace(0, N, num=N) / N - 0.5


def norm(a):
    """Normalise an array by it's maximum value"""
    return a / np.max(np.abs(a))


def text_progress_bar(iteration, num_iteration, max_char = 50):
    """Displays a progress bar with the print function"""
    num_bar = int(np.floor(iteration/num_iteration*max_char)+1)
    num_dot = max_char-num_bar    
    return print('|'*(num_bar) + '.'*(num_dot) + ' %.1f %%'%((iteration+1)/num_iteration*100), end='\r')

def waist_from_nf(radius, angle, wavelength):
    """
    Calculates the Gaussian beam waist parameters from a near field radius and divergence
    """    
    w0 = radius * np.sqrt( ( 1 - np.sqrt( 1 - ( 2*wavelength / ( radius * np.pi * np.tan(angle) ) )**2 ) ) / 2 )
    zr = w0**2*np.pi/wavelength
    z0 = -radius / np.tan(angle)
    return w0, zr, z0

def rolling_window(a, window):
    """
    Reshapes an array to calculate rolling statistics
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_mean(a, window):
    """
    Computes the rolling mean
    """
    return np.nanmean(rolling_window(a, window), axis=-1)

def rolling_std(a, window):
    """
    Computes the rolling standard deviation
    """
    return np.nanstd(rolling_window(a, window), axis=-1)