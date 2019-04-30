import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import comb
import warnings


def wavefront_map(rho, theta, index):
    """Generate a map of the Zernike polynomial, normalised over the unit disk.
    If index is a 1D array, the linear indexing is used.
    If index is a 2D array, the (n, m) indexing is used."""
    # Get (n, m) indices
    if np.size(index) == 1:
        n, m = lin_to_nm(index)
    elif np.size(index) == 2:
        n = index[0]
        m = index[1]
        # Check for correct index
        if np.any(n < 0):
            raise ValueError('n should be positive')
        if np.mod(n, 2) != np.mod(m, 2):
            raise ValueError('n and m should have the same parity')
        if np.abs(m) > n:
            raise ValueError('|m| should be smaller than n')
    else:
        raise ValueError('index should have at most 2 elements')
    # Check input size
    if rho.shape != theta.shape:
        raise ValueError('rho ans theta do not have the same shape')
    # Compute radial polynomial
    R = np.zeros_like(rho)
    for k in np.arange((n - np.abs(m)) / 2 + 1):
        R += (-1)**k * comb(n - k, k) * comb(n - 2 * k,
                                             (n - np.abs(m)) / 2 - k) * rho**(n - 2 * k)
    # Get the full wavefront map
    wave_map = np.sqrt(2 * (n + 1) / (1 + (m == 0))) * R * \
        np.cos(np.abs(m) * theta - (m < 0) * np.pi / 2)
    wave_map[rho > 1] = 0
    return wave_map


def nm_to_lin(n, m):
    """Computes linear index from (n, m) indices"""
    if np.any(n < 0):
        raise ValueError('n should be positive')
    if np.any(np.mod(n, 2) != np.mod(m, 2)):
        raise ValueError('n and m should have the same parity')
    if np.any(np.abs(m) > n):
        raise ValueError('|m| should be smaller than n')
    return n * (n + 1) / 2 + 1 + np.abs(m) - np.mod(m, 2) - np.abs(np.sign(m)) * (1 + np.sign(m) - 2 * np.mod(n, 2)) / 2


def lin_to_nm(j):
    """Computes (n, m) indices from linear index"""
    if np.any(j < 1):
        raise ValueError('j should be strictly positive')
    j -= 1
    n = np.floor((-1 + np.sqrt(1 + 8 * j)) / 2)
    l = j - (n * (n + 1) / 2)
    m = (-1)**(np.mod(l, 2) + np.mod(n, 2) + 1) * \
        (l + np.mod(np.mod(l, 2) + np.mod(n, 2), 2))
    return n, m


def haso_to_nm(j):
    """Computes (n, m) indices from HASO linear index"""
    nm = np.floor(2 * np.sqrt(j) - 1)
    nm += np.mod(nm, 2)
    idxsub = j - (nm / 2)**2
    diffm = np.floor(idxsub / 2)
    m = nm / 2 - diffm
    m *= (-2 * np.mod(idxsub, 2) + 1)
    n = nm / 2 + diffm
    return n, m


def haso_to_lin(j):
    """Computes Phasics linear index from HASO linear index"""
    n, m = haso_to_nm(j)
    return nm_to_lin(n, m)


def nm_to_haso(n, m):
    """Computes HASO linear index from (n, m) indices"""
    if np.any(n < 0):
        raise ValueError('n should be positive')
    if np.any(np.mod(n, 2) != np.mod(m, 2)):
        raise ValueError('n and m should have the same parity')
    if np.any(np.abs(m) > n):
        raise ValueError('|m| should be smaller than n')
    return ((n + np.abs(m)) / 2 + 1)**2 - np.abs(2 * m) + (m < 0) - 1


def lin_to_haso(j):
    """Computes HASO linear index from Phasics linear index"""
    n, m = lin_to_nm(j)
    return nm_to_haso(n, m)


def project(wf_map, N_max=66):
    """
    Project the wavefront map on the Zernike polynomials
    """
    n2, n1 = wf_map.shape
    x = np.linspace(-1, 1, num=n1)
    y = np.linspace(-1, 1, num=n2)
    X, Y = np.meshgrid(x, y)
    rho, theta = cart2pol(X, Y)
    list_z = np.zeros(N_max)
    for k in range(0, N_max):
        z = wavefront_map(rho, theta, k + 1)
        list_z[k] = np.sum(wf_map * z) / np.sum(z * z)
    return list_z


def list_to_map(zernike_list, rho, theta):
    """
    Generate the wavefront map from the Zernike polynomials
    """
    n = zernike_list.size
    wmap = np.zeros_like(rho)
    for k in range(n):
        wmap += zernike_list[k] * wavefront_map(rho, theta, k + 1)
    return wmap


def normalize_list(zernike_list, rms_amplitude):
    """
    Normalizes the zernike coefficients such that
    the rms wavefront amplitude is equal to rms_amplitude.
    """
    return zernike_list * rms_amplitude / np.sqrt(np.sum(zernike_list**2))


def cart2pol(x, y):
    """Convert cartesian to polar coordinates"""
    return np.abs(x + 1j * y), np.angle(x + 1j * y)


def make_focus(wmap, imap, N=1024):
    """
    Generate the Fourier transform from wavefront and intensity maps.
    The resutling focus is scaled by the Fourier limited focus
    """
    n1, n2 = wmap.shape
    if np.mod(wmap.shape, 2).any():
        print('Please use even numbered arrays for Fourier transform')
    padding = N // 2 - np.array(wmap.shape) // 2
    padding[padding < 0] = 0
    padding = [(p,) for p in padding]
    wpad = np.pad(wmap, padding, 'constant')
    ipad = np.pad(imap, padding, 'constant')
    epad = np.sqrt(ipad) * np.exp(1j * 2 * np.pi * wpad)
    efoc = np.abs(np.fft.fftshift(np.fft.fft2(epad)))**2
    eref = np.abs(np.fft.fftshift(np.fft.fft2(np.sqrt(ipad))))**2
    efoc /= np.max(eref)
    return efoc[(N // 2 - n1 // 2):(N // 2 + n1 // 2), (N // 2 - n2 // 2):(N // 2 + n2 // 2)]


def focus_shift_from_zernike(defocus, astig0=None, radBeam=0.035, radDef=None, f=2.034, D=35):
    """
    Calculates focus shift and divergence from defocus and astigmatism 0 degree

    Parameters
    ----------
    defocus: float
        zernike coefficient of defocus, in microns

    astig0: float, optional
        zernike coefficient of astigmatism 0 degree, in microns

    radBeam: float, optional
        beam radius, in metre

    radDef: float, optional
        default radius, to compare measurements with different measured radii, in metre

    f: float, optional
        focal length, in metre

    D: float, optional
        distance from measurement to focussing optic, in metre
    """
    if astig0 is None:
        astig0 = np.zeros_like(defocus)
    if radDef is None:
        radDef = radBeam

    cx = 4 * np.sqrt(3) * (defocus + astig0 / np.sqrt(2)) * 1e-6
    cy = 4 * np.sqrt(3) * (defocus - astig0 / np.sqrt(2)) * 1e-6
    cr = 4 * np.sqrt(3) * defocus * 1e-6

    divergence_x = np.arctan(cx / radBeam**2 * radDef)
    divergence_y = np.arctan(cy / radBeam**2 * radDef)
    divergence_avg = np.arctan(cr / radBeam**2 * radDef)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        foc_shift_x = f**2 / (radBeam / divergence_x + D - f)
        foc_shift_y = f**2 / (radBeam / divergence_y + D - f)
        foc_shift_avg = f**2 / (radBeam / divergence_avg + D - f)

    return foc_shift_avg, foc_shift_x, foc_shift_y, divergence_avg, divergence_x, divergence_y


def zernike_from_focus_shift(dz_x, dz_y=None, radBeam=0.035, f=2.034, D=35):
    """
    Calculates defocus and astigmatism 0 degree (in microns) from X and Y focus shift

    Parameters
    ----------
    dz_x: float
        Focus shift in X, in m

    dz_y: float or None, optional
        focus shift in Y, in m.
        If None, it is the same as dz_x

    radBeam: float, optional
        beam radius in near field, in metre

    f: float, optional
        focal length, in metre

    D: float, optional
        distance from measurement to focussing optic, in metre
    """
    if dz_y is None:
        dz_y = dz_x
    Zx = radBeam / (4 * np.sqrt(3)) * np.tan(radBeam / (f**2 / dz_x - D + f))
    Zy = radBeam / (4 * np.sqrt(3)) * np.tan(radBeam / (f**2 / dz_y - D + f))
    defocus = (Zx + Zy) / 2 * 1e6
    astig0 = (Zx - Zy) * np.sqrt(2) / 2 * 1e6
    return defocus, astig0


def tilt_from_zernike(tilt, radBeam=0.035):
    """Calculate pointing tilt from zernike coefficients 1 and 2

    Parameters:
    -----------
    tilt: float, or float numpy array
        Tilt Zernike coefficient (1,1) and/or (1,-1), in µm

    radBeam: float, optional
        Beam radius, in mm
    """
    return np.arctan(tilt) * 2e-6 / radBeam


zernike_name = ["Piston",
                "Tilt 0deg",
                "Tilt 90deg",
                "Defocus",
                "Astigmatism 0deg",
                "Astigmatism 45deg",
                "Coma 0deg",
                "Coma 90deg",
                "Trefoil 0deg",
                "Trefoil 30deg",
                "1st Spherical ab.",
                "2nd Astigmatism 0deg",
                "2nd Astigmatism 45deg",
                "Tetrafoil 0deg",
                "Tetrafoil 22.5deg",
                "2nd Coma 0deg",
                "2nd Coma 90deg",
                "2nd Trefoil 0deg",
                "2nd Trefoil 30deg",
                "Pentafoil 0deg",
                "Pentafoil 18deg",
                "2nd Spherical ab.",
                "3rd Astigmatism 0deg",
                "3rd Astigmatism 45deg",
                "2nd Tetrafoil 0deg",
                "2nd Tetrafoil 22.5deg",
                "Hexafoil 0deg",
                "Hexafoil 15deg",
                "3rd Coma 0deg",
                "3rd Coma 90deg",
                "3rd Trefoil 0deg",
                "3rd Trefoil 30deg",
                "2nd Pentafoil 0deg",
                "2nd Pentafoil 18deg",
                "Heptafoil 0deg",
                "Heptafoil 12.86deg",
                "3rd Spherical ab.",
                "4th Astigmatism 0deg",
                "4th Astigmatism 45deg",
                "3rd Tetrafoil 0deg",
                "3rd Tetrafoil 22.5deg",
                "2nd Hexafoil 0deg",
                "2nd Hexafoil 15deg",
                "Octafoil 0deg",
                "Octafoil 11.25deg",
                "4th Coma 0deg",
                "4th Coma 90deg",
                "4th Trefoil 0deg",
                "4th Trefoil 30deg",
                "3rd Pentafoil 0deg",
                "3rd Pentafoil 18deg",
                "2nd Heptafoil 0deg",
                "2nd Heptafoil 12.86deg",
                "Nonafoil 0deg",
                "Nonafoil 10deg",
                "4th Spherical ab.",
                "5th Astigmatism 0deg",
                "5th Astigmatism 45deg",
                "4th Tetrafoil 0deg",
                "4th Tetrafoil 22.5deg",
                "3rd Hexafoil 0deg",
                "3rd Hexafoil 15deg",
                "2nd Octafoil 0deg",
                "2nd Octafoil 11.25deg",
                "Decafoil 0deg",
                "Decafoil 9deg"]
