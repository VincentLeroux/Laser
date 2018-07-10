import numpy as np

def prop_two_steps(u1, L1, L2, l, z):
    """
    Fresnel propagator
    Please use SI units

    Inputs:
    ---------
        'u1': input electric field (2D numpy array, float or complex)
        'L1': input plane size (in m)
        'L2': output plane size (in m)
        'l': wavelength (in m)
        'z': propagation distance (in m)

    Output:
    ---------
        'u2': output electric field (2D numpy array, complex)
    """
    M, N = u1.shape
    k = 2*np.pi/l
    
    # Source plane
    dx1 = L1/N
    x1 = np.linspace(0, L1, num=N) - L1/2
    dy1 = L1/M
    y1 = np.linspace(0, L1, num=M) - L1/2
    X1, Y1 = np.meshgrid(x1, y1)
    u = u1 * np.exp( 1j * k / ( 2 * z * L1 ) * ( L1 - L2 ) * ( X1**2 + Y1**2 ) )
    u = np.fft.fftshift(np.fft.fft2(u))
    
    # Dummy (frequency) plane
    fx1 = np.linspace(-0.5, 0.5, num=N)*N/L1
    fy1 = np.linspace(-0.5, 0.5, num=M)*M/L1
    FX1, FY1 = np.meshgrid(fx1, fy1)
    u = np.exp( -1j * np.pi * l * z * L1/L2 * ( FX1**2 + FY1**2 ) ) * u
    u = np.fft.ifft2(np.fft.ifftshift(u))
    
    # Observation plane
    dx2 = L2/N
    x2 = np.linspace(0, L2, num=N) - L2/2
    dy2 = L2/M
    y2 = np.linspace(0, L2, num=M) - L2/2
    X2, Y2 = np.meshgrid(x2, y2)
    u2 = L2/L1 * u * np.exp( -1j * k / ( 2 * z * L2 ) * ( L1 - L2 ) * ( X2**2 + Y2**2 ) )
    u2 *= dx1*dy1/dx2/dy2 # x1 to x2 scale adjustment
    return u2