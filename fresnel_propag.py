import numpy as np

def prop_two_steps(u1, L1, L2, l, z):
    """
    Fresnel propagator
    Please use SI units
    To propagate after the focal plane, first propagate to the focal plane and then out of it
    Also, the near fields should keep the same aspect ratio to the box size ideally...

    Parameters
    ----------
    u1: float or complex 2D np.array
        Input electric field to propagate
    
    L1: float
        Input plane size in meters
    
    L2: float
        Output plane size in meters
    
    l: float
        Wavelength in meters
    
    z: float
        Propagation distance in meters

    Output
    ------
    u2: complex 2D np.array
        Output electric field
    """
    # Input dimensions
    M, N = u1.shape
    # Calculate wave vector amplitude
    k = 2*np.pi/l
    
    # Source plane
    dx1 = L1/N
    x1 = axis_vect(N)*dx1
    dy1 = L1/M
    y1 = axis_vect(M)*dy1
    X1, Y1 = np.meshgrid(x1, y1)
    u = u1 * np.exp( 1j * k / ( 2 * z * L1 ) * ( L1 - L2 ) * ( X1**2 + Y1**2 ) )
    u = np.fft.fftshift(np.fft.fft2(u))
    
    # Dummy (frequency) plane
    fx1 = axis_vect(N)/L1
    fy1 = axis_vect(M)/L1
    FX1, FY1 = np.meshgrid(fx1, fy1)
    u = np.exp( -1j * np.pi * l * z * L1/L2 * ( FX1**2 + FY1**2 ) ) * u
    u = np.fft.ifft2(np.fft.ifftshift(u))
    
    # Observation plane
    dx2 = L2/N
    x2 = axis_vect(N)*dx2
    dy2 = L2/M
    y2 = axis_vect(M)*dy2
    X2, Y2 = np.meshgrid(x2, y2)
    u2 = L2/L1 * u * np.exp( -1j * k / ( 2 * z * L2 ) * ( L1 - L2 ) * ( X2**2 + Y2**2 ) )
    u2 *= dx1*dy1/dx2/dy2 # x1 to x2 scale adjustment
    return u2


def abcd_propag(beampath, profile={'wavelength':800e-9, 'sg_order':6}, num_points = 100, aperture = None, nf_scale=1, ff_scale=1):
    """
    Propagates a beam according to a Beampath object. Assumes that the beam starts in the near field.
    
    Parameters
    ----------
    beampath: Beampath
        Beampath object along which the beam should be propagated.
        The FWHM of the beam will be taken from the initial radius of beampath
    
    profile: dict, optional
        Defines the beam parameters.
        It assumes a centered super-Gaussian beam defined by the keyword 'sg_order',
        and the wavelength defined by the keyword 'wavelength'.
        If one of these two keywords is missing, it will be replaced by the default value and send a warning
    
    num_points: int, optional
        Number of points of the transverse axes
    
    aperture: float 3D np.array, optional
        Aperture of the beam path.
        Values range between 0 (completely blocked) and 1 (completely transparent)
        It should have the same shape as the output efield
    
    nf_scale: float, optional
        Scales the near field box size accordingly.
        By default, the box size is three times the FWHM
    
    ff_scale: float, optional
        Scales the near field box size accordingly.
        By default, the box size is ten times the waist of a Gaussian pilot beam
    
    Outputs
    -------
    efield: complex 3D np.array
        Electric field of the beam at each element of the beampath.
        First dimension is the longitudinal position
        Second and third are the transverse axes
    
    boxsize: float 1D np.array
        Transverse size of the simulation box along the beampath
        
    """
    from abcd import Beampath, Freespace, Thinlens, Interface
    import warnings
    
    # Check inputs
    if not isinstance(beampath, Beampath):
        raise ValueError("First input is not a Beampath object")
        
    # If wavelength not specified, set it at 800 nm and send warning
    if 'wavelength' not in profile.keys():
        profile['wavelength'] = 800e-9
        warnings.warn('Wavelength set at 800 nm')
    # If sg_order not specified, set it at 6 and send warning
    if 'sg_order' not in profile.keys():
        profile['sg_order'] = 6
        warnings.warn('super-Gaussian order set at 6')
            
    # Check how many propagations have to be computed and how many lenses are present
    num_propag = len([1 for elem in beampath.list_elements if isinstance(elem, Freespace)])
    num_lens = len([1 for elem in beampath.list_elements if isinstance(elem, Thinlens)])
    
    
    if aperture is None:
        aperture = np.ones([num_propag+num_lens+1, num_points, num_points])
    
    # Get wave vector
    k = 2*np.pi/profile['wavelength']
    
    # Initialise complex efield array and simulation size
    efield = np.zeros([num_propag+num_lens+1, num_points, num_points])*1j
    boxsize = np.ones([num_propag+num_lens+1])*beampath.radius[0]*6*nf_scale # 3 times the FWHM
    
    # Initialise x and y axes
    x = axis_vect(num_points)/num_points*boxsize[0]
    X,Y = np.meshgrid(x,x)
    
    # Define input electric field
    efield[0,:,:] = gauss2D(X, Y, beampath.radius[0]*2, beampath.radius[0]*2, order=profile['sg_order'])
    # Check for initial divergence and add it if there is one
    if beampath.angle[0]:
        curv = -beampath.radius[0]/np.tan(beampath.angle[0])
        efield[0, :, :] *= np.exp( -1j * k / 2 / curv * ( X**2 + Y**2 ) )
    
    # Add aperture
    efield[0,:,:] *= aperture[0,:,:]
    
    # Start simulation
    for i, elem in enumerate(beampath.list_elements):
        
        # For Thinlens, add focusing curvature
        if isinstance(elem, Thinlens):
            boxsize[i]=boxsize[i-1]
            efield[i, :, :] = efield[i-1, :, :]*np.exp( -1j * k / 2 / elem.focal_length * ( X**2 + Y**2 ) )
            # Add aperture
            efield[i,:,:] *= aperture[i,:,:]
        
        # For Freespace, calculate next boxsize and propagate
        elif isinstance(elem, Freespace):
            # If previous radius is zero, get the one before
            if beampath.radius[i-1]:
                boxsize[i] = boxsize[i-1] * beampath.radius[i]/beampath.radius[i-1]
            else:
                boxsize[i] = boxsize[i-2] * beampath.radius[i]/beampath.radius[i-2]
            # If next boxsize smaller than the wavelength (somewhere in the focus), change scaling
            if np.abs(boxsize[i]) < profile['wavelength']:
                w0, zR, z0 = waist_from_nf(beampath.radius[i-1], beampath.angle[i-1], profile['wavelength'])
                boxsize[i] = np.sign(boxsize[i-1])*10*w0*ff_scale
            
            # Update x and y axes
            x = axis_vect(num_points)/num_points*boxsize[i]
            X,Y = np.meshgrid(x,x)
            # Propagate
            efield[i, :, :] = prop_two_steps(efield[i-1, :, :], boxsize[i-1],
                                             boxsize[i], profile['wavelength'], elem.distance)
            # Add aperture
            efield[i,:,:] *= aperture[i,:,:]
            
        elif isinstance(elem, Interface):
            raise ValueError('Interfaces are not yet implemented....')
        
    return efield, boxsize


def axis_vect(N):
    """
    Returns a centered array.
    If N is odd, x spans from -(N-1)/2 to +(N-1)/2. Otherwise, it spans from -N/2 to +N/2-1
    
    Parameters
    ----------
    N: int
        Number of points
    """
    if N%2:
        x = np.linspace(-(N-1)/2, (N-1)/2,  N)
    else:
        x = np.linspace(-N/2, N/2,  N+1)[:-1:]
    return x

def gauss2D(x, y, fwhmx, fwhmy, x0=0, y0 = 0, offset=0, order=1, int_FWHM = True):
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
    return np.exp(-np.log(2)*coeff*((2*(x-x0)/fwhmx)**2 + (2*(y-y0)/fwhmy)**2)**order) + offset

def waist_from_nf(radius, angle, wavelength):
    """
    Calculates the Gaussian beam waist parameters from a near field radius and divergence
    
    Parameters
    ----------
    radius: float
        Near field beam radius
    
    angle: float
        Near field beam half-divergence
    
    wavelength: float
        Beam wavelength
    
    Outputs
    -------
    w0: float
        Beam waist in focus
    
    zr: float
        Rayleigh length
    
    z0: float
        Waist position from input near field
    """
    w0 = radius * np.sqrt( ( 1 - np.sqrt( 1 - ( 2*wavelength / ( radius * np.pi * np.tan(angle) ) )**2 ) ) / 2 )
    zr = w0**2*np.pi/wavelength
    z0 = -radius / np.tan(angle)
    return w0, zr, z0