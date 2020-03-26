import numpy as np

def n2_sapphire (l):
    '''
    Returns the nonlinear index of sapphire in m2/W,
    valid between 500 and 1500 nm.
    
    Parameters:
    ===========
    l: float
        Wavelength in meters
    '''
    l_nm = l*1e9
    n2_0 = 2.5 # 1e-16 cm2/W
    l0 = 266 # nm
    N1 = 2.3 # 1e-16 cm2/W
    l1 = 46.6 # nm
    N2 = 1.0 # 1e-16 cm2/W
    l2 = 1086.3 # nm
    return (n2_0 + N1 * np.exp( -( l_nm - l0 ) / l1 ) + N2 * np.exp( -( l_nm - l0 ) / l2 ))*1e-20

def sellmeier_equation(l, B1, B2, B3, C1, C2, C3):
    '''
    Returns the refractive index, according to Sellmeier coefficients
    
    Parameters:
    ===========
    l: float
        Wavelength, in um
    B1, B2, B3: floats
        Numerator coefficients, no units
    C1, C2, C3: floats
        Denominator coefficients, in um2
    '''
    return np.sqrt(1 + B1 * l**2 / (l**2 - C1) + B2 * l**2 / (l**2 - C2) + B3 * l**2 / (l**2 - C3))

def n_fused_silica(l):
    '''
    Returns the linear index of fused silica.
    
    Parameters:
    ===========
    l: float
        Wavelength in meters
    '''
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2
    return sellmeier_equation(l*1e6, B1, B2, B3, C1, C2, C3)

def n_sapphire(l):
    '''
    Returns the linear index of sapphire.
    
    Parameters:
    ===========
    l_nm: float
        Wavelength in meters
    '''
    B1 = 1.4313493
    B2 = 0.65054713
    B3 = 5.3414021
    C1 = 0.07266313**2
    C2 = 0.1193242**2
    C3 = 18.028251**2
    return sellmeier_equation(l*1e6, B1, B2, B3, C1, C2, C3)
    
def n_bk7(l):
    '''
    Returns the linear index of BK7.
    
    Parameters:
    ===========
    l_nm: float
        Wavelength in meters
    '''
    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945
    C1 = 0.00600069867
    C2 = 0.0200179144
    C3 = 103.560653
    return sellmeier_equation(l*1e6, B1, B2, B3, C1, C2, C3)

def ne_sapphire(l):
    '''
    Returns the linear index of sapphire along the extraodinary axis.
    
    Parameters:
    ===========
    l_nm: float
        Wavelength in meters
    '''
    B1 = 2.63585614
    B2 = -0.581713315
    B3 = 6.28436363
    C1 = 0.09033847**2
    C2 = 0.090329096**2
    C3 = 19.398880**2
    return sellmeier_equation(l*1e6, B1, B2, B3, C1, C2, C3)