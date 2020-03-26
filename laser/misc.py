import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
import matplotlib.image as mpimg


def get_moments(image):
    """
    Compute image centroid and statistical waist from the intensity distribution.

    Parameters:
    -----------
    image: 2D numpy array
    """
    # Build axes in pixels
    ny, nx = image.shape
    x, y = np.arange(nx), np.arange(ny)
    X, Y = np.meshgrid(x, y)
    # Zeroth moment
    c0 = np.sum(image)
    # First moments
    cx = np.sum(X * image) / c0
    cy = np.sum(Y * image) / c0
    # Second centered moments
    sx2 = np.sum((X - cx)**2 * image) / c0
    sy2 = np.sum((Y - cy)**2 * image) / c0
    return cx, cy, 2 * np.sqrt(sx2), 2 * np.sqrt(sy2)


def get_encircled_energy(image, center="geometric"):
    """
    Compute the encircled energy of an intensity distribution

    Parameters
    ----------
    image: 2D numpy array
        Intensity distribution

    center: {"geometric", "centroid", "peak"} or tuple, optional
        Defines from which point is the encircled energy calculated.
    """
    # Get the center position
    if center == "geometric":
        mask = np.zeros_like(image)
        mask[image>=(image.max()/3)] = 1
        cx, cy, _, _ = get_moments(mask)
    elif center == "centroid":
        cx, cy, _, _ = get_moments(image)
    elif center == "peak":
        cy, cx = np.unravel_index(np.argmax(image), image.shape)
    else:
        cx, cy = center[0], center[1]

    # build radius axis
    ny, nx = image.shape
    x, y = np.arange(nx), np.arange(ny)
    Xc, Yc = np.meshgrid(x - cx, y - cy)
    R, _ = cart2pol(Xc, Yc)

    # Sort the radius and get the index
    idx_sort = np.argsort(R, axis=None)
    rad_sort = R.ravel()[idx_sort]

    # Get the encircled energy
    en_circ = np.cumsum(image.ravel()[idx_sort])
    en_circ = np.insert(en_circ, 0, 0.0) / np.sum(image)
    rad_sort = np.insert(rad_sort, 0, 0.0)

    return rad_sort, en_circ


def get_fwhm(intensity, interpolation_factor=1, kind='cubic'):
    """
    Get the Full Width at Half Maximum of the 1D intensity distribution

    Parameters
    ----------
    intensity: 1D numpy array
        intensity distribution

    interpolation_factor: int, optional
        Interpolate the data for a more accurate calculation
    """
    position = np.arange(intensity.size)
    pos_i = np.linspace(np.min(position), np.max(position),
                        interpolation_factor * position.size)
    inten_i = interp1d(position[:], intensity[:], kind=kind)
    idx = (inten_i(pos_i) >= np.max(inten_i(pos_i)) * 0.5).nonzero()[0]
    return pos_i[idx[-1] + 1] - pos_i[idx[0]]


def gauss2D(x, y, fwhmx, fwhmy, x0=0, y0=0, offset=0, order=1, int_FWHM=True):
    """
    Define a (super-)Gaussian 2D beam.

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
    Define a (super-)Gaussian 1D beam. Identical to laser.misc.gauss2D but in 1D.

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
    return np.real(r * np.exp(1j * theta)), np.imag(r * np.exp(1j * theta))


def array_trim(ar):
    """Trim zeros of 2D map"""
    ar_trim = ar.copy()
    ar_trim = ar_trim[:, ar_trim.any(axis=0)]  # trim columns
    ar_trim = ar_trim[ar_trim.any(axis=1), :]  # trim rows
    return ar_trim


def vect(N):
    """Return a centered array between -0.5 and 0.5"""
    return np.linspace(0, N, num=N) / N - 0.5


def norm(a):
    """Normalise an array by it's maximum value"""
    return a / np.max(np.abs(a))


def text_progress_bar(iteration, num_iteration, max_char=50):
    """Display a progress bar with the print function"""
    num_bar = int(np.floor(iteration / num_iteration * max_char) + 1)
    num_dot = max_char - num_bar - 1
    return print('|' * (num_bar) + '.' * (num_dot) + ' %.1f %%' % ((iteration + 1) / num_iteration * 100), end='\r')


def waist_from_nf(radius, angle, wavelength):
    """
    Calculate the Gaussian beam waist parameters from a near field radius and divergence
    """
    w0 = radius * \
        np.sqrt(
            (1 - np.sqrt(1 - (2 * wavelength / (radius * np.pi * np.tan(angle)))**2)) / 2)
    zr = w0**2 * np.pi / wavelength
    z0 = -radius / np.tan(angle)
    return w0, zr, z0


def rolling_window(a, window):
    """
    Reshape an array to calculate rolling statistics
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_mean(a, window):
    """
    Compute the rolling mean
    """
    return np.nanmean(rolling_window(a, window), axis=-1)


def rolling_std(a, window):
    """
    Compute the rolling standard deviation
    """
    return np.nanstd(rolling_window(a, window), axis=-1)


def moving_average(a, window):
    """
    Very fast moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def add_noise(image, density=None, amplitude=1, kind='quintic', seed=None):
    """
    Add noise to a 2D numpy array. If "density" is specified, the noise is interpolated to have smooth variations.

    Parameters
    ----------
    image: 2D numpy.array
        Image on which the noise should be added

    density: int, 2-tuple, optional
        Noise density. if equal to the image size, equivalent to "None"

    amplitude: float, optional
        Amplitude of the noise. If "1", image is modulated by +- 100%

    kind: {'linear', 'cubic', 'quintic'}
        Type of 2D-interpolation. 'linear' can be used but it is pretty ugly.

    seed: int, optional
        Seed for random number generation        
    """
    ny, nx = image.shape
    if density is None:
        density = (nx, ny)
    try:
        dx = density[0]
        dy = density[1]
    except TypeError:
        dx = density
        dy = density
    np.random.seed(seed)
    noise_raw = np.random.rand(int(dy), int(dx))
    x_raw = np.arange(int(dx))
    y_raw = np.arange(int(dy))
    noisefunc = interp2d(x_raw, y_raw, noise_raw, kind=kind)
    x = np.linspace(np.min(x_raw), np.max(x_raw), nx)
    y = np.linspace(np.min(y_raw), np.max(y_raw), ny)
    noise = noisefunc(x, y)
    noise = (noise - np.min(noise)) / np.ptp(noise) * 2 - 1

    image_noise = image * (1 + amplitude * noise) / (1 + amplitude)
    image_noise *= np.sum(image) / np.sum(image_noise)
    return image_noise

def RGB_image_to_grayscale(image_path, reverse_scale=True, crop=None, downsample=None):
    """
    Convert RGB colors to lightness grayscale
    
    Parameters:
    ===========
    image_path: str
        location of the image to import and convert to greyscale
    
    reverse_scale: boolean, optional
        choose to flip the lightness scale or not. Stays between 0 and 1
        
    crop: None or 4-tuple, optional
        Limits to crop the image
       
    downsample: None or int, optional
        Downsample the data by the given amount. Currently the downsampling is
        done by selecting data with a period given by this parameter
    """
    im_rgb = np.float64(mpimg.imread(image_path))/255
    if crop is not None:
        im_rgb = im_rgb[crop[0]:crop[1],crop[2]:crop[3]]
    if downsample is not None:
        im_rgb = im_rgb[::downsample,::downsample]
    M = np.array([[0.412453,0.357580,0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])
    im_xyz = (M[None,None,:]@im_rgb[:,:,:,None]).squeeze()
    L = np.zeros_like(im_xyz[:,:,0])
    select = im_xyz[:,:,1]>(6/29)**3
    L[select] = 116*im_xyz[select,1]**(1/3)-16
    L[~select] = (29/3)**3*im_xyz[~select,1]
    L /= 100
    if reverse_scale:
        L = 1-L
    return L

def norm_minmax(a):
    """
    Normalize the data by setting the minimum at 0 and the maximum at 1.
    
    Parameters:
    ===========
    a: numpy.array
        Data to normalize
    """
    return (a-a.min())/(a.max()-a.min())

def get_ellipse_moments(image, dx=1, dy=1, cut=None):
    """
    Compute the moments of the beam profile and give the ellipse parameters.
    
    Parameters:
    ===========
    image: 2D numpy.array
        Intensity profile of the data
    
    dx: float, optional
        Step of the horizontal axis. Defaults to 1
    
    dy: float, optional
        Step of the vertical axis. Defaults to 1
    
    cut: None or float, optional
        Threshold below which the data is ignored
    
    Outputs:
    ========
    cx: float
        Horizontal position of the center of mass
    
    cy: float
        Vertical position of the center of mass
    
    rx: float
        Radius of the ellipse close to the horizontal axis
    
    ry: float
        Radius of the ellipse close to the vertical axis
    
    theta: float
        Angle of the ellipse from the horizontal axis
    
    gamma: float
        If gamma = 1, rx is the major axis.
        If gamma = -1, ry is the major axis.
    """
    im = image.copy()
    if cut is not None:
        im -= cut
        im[im<0]=0
    # Build axes in pixels
    ny, nx = im.shape
    x, y = np.arange(nx), np.arange(ny)
    X, Y = np.meshgrid(x, y)
    # Zeroth moment
    c0 = np.sum(im)
    # First moments
    cx = np.sum(X * im) / c0
    cy = np.sum(Y * im) / c0
    # Second centered moments
    sx2 = np.sum((X - cx)**2 * im) / c0
    sy2 = np.sum((Y - cy)**2 * im) / c0
    sxy = np.sum((Y - cy) * (X - cx) * im) / c0
    # Derived quantities
    gamma = np.sign(sx2-sy2)
    cor_term = gamma * np.sqrt((sx2 - sy2)**2 + 4 * sxy**2)
    rx = np.sqrt(2 * ( sx2 + sy2 + cor_term ))
    ry = np.sqrt(2 * ( sx2 + sy2 - cor_term ))
    theta = 0.5 * np.arctan(2 * sxy / (sx2 - sy2))
    cx *= dx
    cy *= dy
    rx *= dx
    ry *= dy
    return cx, cy, rx, ry, theta, gamma

def biquad(X, c, x1, y1, x2, y2, xy):
    """
    Biquadratic surface, for curve fitting
    """
    x,y = X
    return x2*x**2 + y2*y**2 + xy*x*y + x1*x + y1*y + c

def bilin(X, c, x1, y1):
    """
    Bilinear surface, for curve fitting
    """
    x,y = X
    return x1*x + y1*y + c

def remove_baseline(image, threshold, quadratic=True):
    """
    Fit the baseline of a 2D image and removes it from this image.
    
    Parameters:
    ===========
    image: 2D numpy.array
        Intensity profile of the data
        
    threshold: float
        Threshold below which the data is considered for the baseline fit
    
    quadratic: boolean, optional
        If True, a biquadratic fit is used to calculate the baseline.
        If False, a bilinear fit is used.
    """
    ny,nx = image.shape
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    X,Y = np.meshgrid(x,y)
    select = image<threshold
    base_data = image[select]
    Xb, Yb = X[select], Y[select]
    if quadratic:
        c, x1, y1, x2, y2, xy = curve_fit(biquad, (Xb, Yb), base_data, p0=[0]*6)[0]
        baseline = x2*X**2 + y2*Y**2 + xy*X*Y + x1*X + y1*Y + c
    else:
        c, x1, y1 = curve_fit(bilin, (Xb, Yb), base_data, p0=[0]*3)[0]
        baseline = x1*X + y1*Y + c
    return (image - baseline)
    
def dx (x):
    return np.mean(np.diff(x))

def polygauss(x, y, fwhmx, fwhmy, x0=0, y0=0, theta=0, offset=0, order=1, polygon=None, angle=0, int_FWHM=True):
    """
    Returns a 2D (super-)Gaussian beam with a polygonal geometry.
    """
    xr = np.zeros_like(x)
    yr = np.zeros_like(y)
    xr = np.cos(theta)*x - np.sin(theta)*y
    yr = np.sin(theta)*x + np.cos(theta)*y
    coeff = 1.0
    if int_FWHM:
        coeff = 0.5
    profile = np.zeros_like(x)
    
    if polygon is None or polygon < 5:
        profile = np.exp(-np.log(2) * coeff * ((2 * (xr - x0) / fwhmx)**2 + (2 * (yr - y0) / fwhmy)**2)**order) + offset
    else:
        axes = np.linspace(0,2*np.pi, num=polygon, endpoint=False)
        Xp = np.cos(axes[None,None,:]+angle)*(xr[:,:,None]-x0)/fwhmx + np.sin(axes[None,None,:]+angle)*(yr[:,:,None]-y0)/fwhmy
        Xp[Xp<0]=0
        profile = np.exp(-np.log(2) * coeff * np.sum( (2*Xp)**(2*order) , axis=-1)) + offset
    return profile
    
def change_sigma_def(sigma_in, level_in, level_out, order_sg):
    """
    Calculate the waist at 'level_out' of the max from the waist at 'level_in' for a (super-)Gaussian beam. For example change_sigma_def(10, 0.5, 0.1, 4) gives the full width at 10% for a 4th order super_Gaussian with a FWHM of 10.
    """
    return sigma_in * (np.log(level_out)/np.log(level_in))**(1/order_sg)
