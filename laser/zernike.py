import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def zernike_map(rho, theta, index):
    """Generate a map of the Zernike polynomial, normalised over the unit disk.
    If index is a 1D array, the linear indexing is used.
    If index is a 2D array, the (n, m) indexing is used."""
    # Get (n, m) indices
    if np.size(index) == 1:
        n, m = zernike_lin_to_nm(index)
    elif np.size(index) == 2:
        n = index[0]
        m = index[1]
        # Check for correct index
        if np.any(n < 0):
            raise ValueError('n should be positive')
        if np.mod(n,2) != np.mod(m, 2):
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
    for k in np.arange((n-np.abs(m))/2+1):
        R += (-1)**k*comb(n-k, k)*comb(n-2*k, (n-np.abs(m))/2-k)*rho**(n-2*k)
    # Get the full wavefront map
    wave_map = np.sqrt(2*(n+1)/(1+(m==0)))*R*np.cos(np.abs(m)*theta - (m<0)*np.pi/2)
    wave_map[rho>1]=0
    return wave_map

def zernike_nm_to_lin(n, m):
    """Computes linear index from (n, m) indices"""
    if np.any(n < 0):
        raise ValueError('n should be positive')
    if np.any(np.mod(n,2) != np.mod(m, 2)):
        raise ValueError('n and m should have the same parity')
    if np.any(np.abs(m) > n):
        raise ValueError('|m| should be smaller than n')
    return n*(n+1)/2 + 1 + np.abs(m) - np.mod(m, 2) - np.abs(np.sign(m))*(1 + np.sign(m) - 2*np.mod(n, 2))/2

def zernike_lin_to_nm(j):
    """Computes (n, m) indices from linear index"""
    if np.any(j < 1):
        raise ValueError('j should be strictly positive')
    j -= 1
    n = np.floor((-1 + np.sqrt(1+8*j))/2)
    l = j - (n*(n+1)/2)
    m = (-1)**(np.mod(l, 2) + np.mod(n, 2) + 1)*(l + np.mod(np.mod(l, 2) + np.mod(n, 2), 2))
    return n, m

def zernike_project(wf_map, N_max = 66):
    """
    Project the wavefront map on the Zernike polynomials
    """
    n2,n1 = wf_map.shape
    x = np.linspace(-1,1,num=n1)
    y = np.linspace(-1,1,num=n2)
    X, Y = np.meshgrid(x,y)
    rho, theta = cart2pol(X,Y)
    list_z = np.zeros(N_max)
    for k in range(0,N_max):
        z = zernike_map(rho, theta, k+1)
        list_z[k] = np.sum(wf_map*z)/np.sum(z*z)
    return list_z

def zernike_list_to_map(zernike_list, rho, theta):
    """
    Generate the wavefront map from the Zernike polynomials
    """
    n = zernike_list.size
    wmap = np.zeros_like(rho)
    for k in range(n):
        wmap += zernike_list[k]*zernike_map(rho, theta, k+1)
    return wmap

def cart2pol(x,y):
    """Convert cartesian to polar coordinates"""
    return np.abs(x+1j*y), np.angle(x+1j*y)
