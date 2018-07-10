import glob
import numpy as np
import scipy.constants as cst
from scipy.interpolate import interp1d

def laser_gain_step(J_in, g_in):
    """
    Computes one iteration of Frantz-Nodvik gain simulation,
    and returns the output normalised fluence and the gain left 
    in the crystal.
    J is a number, g is a number
    """
    J_out = np.log( np.exp(g_in) * (  np.exp(J_in)-1 ) +1 )
    g_left = g_in - ( J_out - J_in )
    return J_out, g_left

def laser_gain_crystal_length(J_in, g_in):
    """
    Computes the laser gain over the length of the crystal, and returns 
    the output normalised fluence and the gain left in the crystal.
    J is a number, g is a 1D array
    """
    g_left = np.zeros_like(g_in)
    J_out = np.copy(J_in)
    
    for idx in np.arange(g_in.size):
        J_out, g_left[idx] = laser_gain_step(J_out, g_in[idx])
    
    return J_out, g_left

def laser_gain_crystal_section(J_in, g_in):
    """
    Computes the laser gain over the length and spatial profile of the crystal,
    and returns the output normalised fluence and the gain left in the crystal.
    J is a 2D array, g is a 3D array
    """
    n3, n2, n1 = np.shape(g_in)
    J_out = np.copy(J_in.flatten())
    g_left = np.copy(g_in.reshape(n3, n1*n2))
    
    for idx in np.arange(n1*n2):
        J_out[idx], g_left[:,idx] = laser_gain_crystal_length(J_out[idx], g_left[:,idx])
    
    J_out = J_out.reshape(n2, n1)
    g_left = g_left.reshape(n3, n2, n1)
    return J_out, g_left

def laser_gain_chirped_pass(F_in, lambda_, g_in):
    """
    Computes the laser gain over the length and spatial profile of the crystal 
    for the laser spectrum,and returns the output normalised fluence
    and the gain left in the crystal.
    F is a 3D array, g is a 3D array, lambda_ is a 1D array
    """
    F_in, lambda_, g_in = check_input_dimensions(F_in, lambda_, g_in)
    sigma = gain_cross_section_tisa(lambda_.flatten()) # m^2
    F_sat = cst.h*cst.c/lambda_.flatten()/sigma # J/m^2
    n4, n2, n1 = F_in.shape
    J_out = np.zeros_like(F_in)
    g_left = np.copy(g_in)
    
    for idx in np.arange(n4):
        J_out[idx,:,:], g_left = laser_gain_crystal_section(F_in[idx,:,:]/F_sat[idx], g_left)
    
    F_out = J_out*np.tile(F_sat.reshape(n4,1,1), (1,n2,n1))
    return F_out, g_left

def laser_gain_chirped_bowtie(F_in, lambda_, g_in, Npass=1, Tpass=1.0):
    """
    Compute the laser gain of a bowtie amplifier:
    the laser profile is flipped left-right at each pass
    F is a 3D array, g is a 3D array, lambda_ is a 1D array
    """
    F_out = F_in.copy()
    g_left = g_in.copy()
    for idx in np.arange(Npass):
        F_out, g_left = laser_gain_chirped_pass(F_out, lambda_, g_left)
        g_left = np.flip(np.flip(g_left, 0),2)
        F_out = F_out*Tpass
    return F_out, g_left

def gain_cross_section_tisa(lambda_):
    """
    Returns the gain cross section of titanium-sapphire in m^2
    in function of the wavelength in m
    """
    filename = glob.glob('./**/gain_cross_section.txt', recursive=True)
    data = np.genfromtxt(filename[0],skip_header=1, delimiter='; ')
    l = data[:,0] # in nm
    s = data[:,1] # in cm^2
    cross_section = interp1d(l, s, kind='cubic', bounds_error=False, fill_value=0)
    return cross_section(lambda_*1e9)*1e-4

def abs_cross_section_tisa(lambda_):
    """
    Returns the absorption cross section of titanium-sapphire in m^2
    in function of the wavelength in m
    """
    filename = glob.glob('./**/absorption_cross_section.txt', recursive=True)
    data = np.genfromtxt(filename[0],skip_header=1, delimiter='; ')
    l = data[:,0] # in nm
    s = data[:,1] # in cm^2
    cross_section = interp1d(l, s, kind='cubic', bounds_error=False, fill_value=0)
    return cross_section(lambda_*1e9)*1e-4

#class InputError(Exception):
    """
    Error class to deal with input dimension errors
    """
 #   pass

def check_input_dimensions(F_in, lambda_, g_in):
    """Handle input dimensions, reshape inputs, and throw errors"""
    if lambda_.ndim != 1:
        raise InputError('lambda_ should be a 1D array')
    if g_in.ndim > 3:
        raise InputError('g_in should be a 3D array')
    if F_in.ndim > 3:
        raise InputError('F_in should be a 3D array')
    
    if F_in.ndim == 1: # No spatial dimension
        if g_in.ndim == 1: # Check is gain also has no spatial dimension
            n3 = g_in.size
            g_in = g_in.reshape(n3,1,1)
        else:
            raise InputError('F_in and g_in spatial dimensions do not match')
        n4 = F_in.size
        if lambda_.size != n4: # Check if same spectral size
            raise InputError('F_in and lambda_ do not have the same spectral size')
        F_in = F_in.reshape(n4,1,1) # Reshape in expected 3D array
        
    elif F_in.ndim == 2: # No spectral dimension
        if lambda_.size != 1: # lambda_ should be a number
            raise InputError('F_in and lambda_ do not have the same spectral size')
        n2, n1 = F_in.shape
        if g_in.ndim < 2: # check g_in number of dimensions
            raise InputError('F_in and g_in spatial dimensions do not match')
        elif g_in.ndim == 2: # if g_in has no longitudinal dimension specified
            ng2, ng1 = g_in.shape
            if ng1 != n1 or ng2 != n2:
                raise InputError('F_in and g_in spatial dimensions do not match')
            g_in = g_in.reshape(1,n2,n1)
        elif g_in.ndim == 3: # if g_in has a longitudinal dimension
            ng3, ng2, ng1 = g_in.shape
            if ng1 != n1 or ng2 != n2:
                raise InputError('F_in and g_in spatial dimensions do not match')
        F_in = F_in.reshape(1,n2,n1)
        
    elif F_in.ndim == 3: # Spatial and spectral dimensions
        n4, n2, n1 = F_in.shape
        if lambda_.size != n4: # Check if same spectral size
            raise InputError('F_in and lambda_ do not have the same spectral size')
        if g_in.ndim < 2: # check g_in number of dimensions
            raise InputError('F_in and g_in spatial dimensions do not match')
        elif g_in.ndim == 2: # if g_in has no longitudinal dimension specified
            ng2, ng1 = g_in.shape
            if ng1 != n1 or ng2 != n2:
                raise InputError('F_in and g_in spatial dimensions do not match')
            g_in = g_in.reshape(1,n2,n1)
        elif g_in.ndim == 3: # if g_in has a longitudinal dimension
            ng3, ng2, ng1 = g_in.shape
            if ng1 != n1 or ng2 != n2:
                raise InputError('F_in and g_in spatial dimensions do not match')
    return F_in, lambda_, g_in
