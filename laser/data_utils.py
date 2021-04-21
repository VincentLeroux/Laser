import numpy as np
import matplotlib.pyplot as plt
from laser.misc import polygauss, add_noise, biquad, norm_minmax, remove_baseline, get_ellipse_moments, get_fwhm, get_moments, cart2pol, dx, gauss2D
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import interp1d
from laser.plot_utils import cmap_nicify

def load_oceanoptics_spectra(filename):
    """
    Import Spectrum data from OCean optics text file
    Oldest spectrum in position -1; youngest in position 0.
    
    Parameters
    ----------
    filename: str
        Location of file
    """
    try:
        data_raw = np.loadtxt(filename)
    except ValueError:
        data_raw = np.loadtxt(filename, dtype=str)
        N = data_raw.shape[0]
        data_raw = [np.float64(s.replace(',','.')) for s in data_raw.flatten()]
        data_raw = np.reshape(data_raw, [N, len(data_raw)//N])
    return data_raw


def random_beam_generator(N):
    """
    Generate a beam profile with a random position, size,
    ellipticity, super-Gaussian order, with background noise
    and quadratic offset, with random modulations of random
    frequency of the intensity. Used mainly to test beam analysis scripts.
    
    Parameters
    ----------
    N: int
        Size of output nd.array (N x N)
    """
    x = np.linspace(-1,1,N)
    X,Y = np.meshgrid(x,x)
    fwhmx = (np.random.rand()/0.3+0.1)/2
    fwhmy = (np.random.rand()/0.3+0.1)/2
    x0=(np.random.rand()-0.5)/2
    y0=(np.random.rand()-0.5)/2
    theta=np.random.rand()*2*np.pi
    order=np.random.randint(1,10)
    polygon=np.random.randint(1,10)
    angle=np.random.rand()*2*np.pi
    p = polygauss(X,Y,fwhmx,fwhmy,x0,y0,theta, order=order, polygon=polygon, angle=angle)
    p = add_noise(p, density=N/(np.random.rand()*10+1), amplitude=np.random.rand()/2+0.5)
    
    c = np.random.rand()
    x1 = np.random.rand()-0.5
    y1 = np.random.rand()-0.5
    x2 = np.random.rand()-0.5
    y2 = np.random.rand()-0.5
    xy = np.random.rand()-0.5
    b = biquad((X,Y), c, x1, y1, x2, y2, xy)
    b = np.random.rand()*norm_minmax(b)
    b += np.random.rand()**2*np.random.rand(*b.shape)
    return p+b
    
    
def dualgauss(x, x1, x2, w1, w2, a1, a2, c=0):
    """
    Sum of two Gaussian distributions. For curve fitting.
    
    Parameters
    ----------
    x: np.array
        Axis
        
    x1: float
        Center of 1st Gaussian curve
        
    x2: float
        Center of 2nd Gaussian curve
        
    w1: float
        Width of 1st Gaussian curve
        
    w2: float
        Width of 2nd Gaussian curve
        
    a1: float
        Amplitude of 1st Gaussian curve
        
    a2: float
        Amplitude of 2nd Gaussian curve
    
    c: float, optional
        Offset, defaults to 0
    """
    return a1*np.exp(-0.5*((x-x1)/w1)**2)+a2*np.exp(-0.5*((x-x2)/w2)**2) + c

    
def beam_analysis (image, x=None, y=None, baseline=None, plot=True, unit='mm', tilt_cutoff=None, beam_energy=None, unit_en = 'J', threshold_fluence=None, threshold_fluence2=None, quadratic=False, th_crop=0.2, scale_crop=1.5, threshold_hot_pixel=None):
    """
    Automatic analysis of laser beam intensity profile.
    
    Parameters
    ----------
    image: 2D np.array
        Image to analyze
        
    x: None or 1D np.array, optional
        Horizontal axis. If None, a np.arrange is used.
    
    y: None or 1D np.array, optional
        Vertical axis. If None, a np.arrange is used.
    
    baseline: None or float, optional
        if float, threshold for the removal of the background baseline.
        if None (default), the threshold is automatically computed from the
        histogram distribution of the pixel counts. The background noise
        is fitted by a Gaussian distribution and the threshold is set at
        the mean + 2 sigma.
    
    plot: bool, optional
        If True (default), the function 'beam_analysis_plot' is called
    
    unit: str, optional
        Unit of the space axis. Default is 'mm'.
    
    tilt_cutoff: None or float, optional
        Threshold for the calculation of the tilt of the intensity profile plateau.
        If None (default), the threshold is 1/3 of the plateau average values
    
    beam_energy: None or float, optional
        Energy contained within the image, with the unit defined by unit_en.
        If None (default), the output is in arbitrary units.
        If float, the fluence is calculated in unit_en/unit^2
        
    unit_en: str, optional
        Unit of the energy. Default is 'J'.
    
    threshold_fluence: None or float, optional
        Lists the data above the fluence given by this parameter. If None (default), it is not computed.
    
    threshold_fluence2: None or float, optional
        Lists the data above the fluence given by this parameter. If None (default), it is not computed.
    
    quadratic: bool, optional
        If True, use a bi-quadratic fit for the background. If False (default), use a bi-linear fit.
    
    th_crop: float, optional
        threshold to compute a rough beam size to restrict the analyzed area. Default is 0.2.
    
    scale_crop: float, optional
        Restricted area around the beam is defined by scale_crop times the rough beam size. Adjust
        this parameter to allow more or less of the background in the analysis. Defaults to 1.5.
    
    threshold_hot_pixel: None or float, optional
        If a float is given (typically 3), a pixel value that is too large compared to its neighbors
        will be replaced by the average of its neighbors. threshold_hot_pixel sets the sensitivity of this
        detection. If None, no filtering is applied.
    
    Output
    ------
    output_dict: dict
        dictionary containing all the computed data:
            * center_x: float, horizontal position of the centroid
            
            * center_y: float, vertical position of the centroid
            
            * radius_x: float, radius of the beam close to the
                        horizontal axis (in the beam reference frame)
            
            * radius_y: float, radius of the beam close to the
                        vertical axis (in the beam reference frame)
            
            * theta_ell: float, angle of the beam major axis
            
            * gamma_ell: {-1,1}, if 1, radius_x is the major axis;
                         if -1, radius_y is the major axis.
            
            * im_filter: 2D np.array, scaled image with the baseline removed
            
            * baseline: float, threshold for the removal of the background baseline.
            
            * flatness_axis: 1D np.array, azimuthal axis
            
            * flatness_data: 1D np.array, asymmetry of the intensity plateau
            
            * top_mean: float, mean value of the plateau
            
            * top_rms: float, rms fluctuation of the plateau, relative to the mean
            
            * top_ptp: float, peak-to-peak amplitude of the plateau, relative to the mean
            
            * fit_x: np.array, result of super-Gaussian fit of the horizontal lineout,
                     contains [center, fwhm, order, amplitude]. The order is defined as x^(2*order)
            
            * fit_y: np.array, result of super-Gaussian fit of the vertical lineout,
                     contains [center, fwhm, order, amplitude]. The order is defined as x^(2*order)
            
            * energy: float or None, energy contained within the image.
            
            * unit_en: str, unit of the energy value
    """
    # Remove hot pixels
    if threshold_hot_pixel is not None:
        image = np.float64(image)
        mask = np.ones((3,3))
        mask[1,1]=0
        imm = median_filter(image, footprint=mask, mode='nearest')
        diff = image-imm
        image[np.abs(diff)>np.std(diff)*threshold_hot_pixel] = imm[np.abs(diff)>np.std(diff)*threshold_hot_pixel]
    
    # Normalize image
    imc = norm_minmax(image)
    # Remove baseline
    if baseline is None:
        # Automatic baseline detection: fit of noise distribution,
        # baseline at mean + 2 sigma of noise distribution
        try:
            yh, xh = np.histogram(imc.ravel(), bins=int(np.sqrt(np.maximum(*imc.shape))))
            xh = xh[:-1]
            cf = curve_fit(dualgauss, xh, yh,
                           p0=[xh[np.argmax(yh[xh<0.1])],
                           xh[xh[xh<0.1].size + np.argmax(yh[xh>0.1])],
                           get_fwhm(yh)*dx(xh)/2,np.mean(xh)/5,
                           np.max(yh[xh<0.1]),np.max(yh[xh>0.1]), 0],
                           bounds=([0]*7, [1,1,1,1,np.inf,np.inf,np.inf]))[0]
            idx = np.argmax(cf[-2:])
            baseline = cf[0+idx] + 2*np.abs(cf[2+idx])
        except RuntimeError:
            print('Automatic baseline not found, defaults to 20%')
            baseline=0.2
    imc = remove_baseline(imc, baseline, quadratic=quadratic)
    
    # Cut image around beam
    mask = np.zeros_like(imc)
    mask[imc>th_crop]=1
    cx_rough, cy_rough, rx_rough, ry_rough,_,_ = get_ellipse_moments(mask)
    r_rough = np.sqrt(rx_rough*ry_rough)
    ym = int(np.maximum(cy_rough-scale_crop*r_rough, 0))
    yp = int(np.minimum(cy_rough+scale_crop*r_rough, imc.shape[0]))
    xm = int(np.maximum(cx_rough-scale_crop*r_rough, 0))
    xp = int(np.minimum(cx_rough+scale_crop*r_rough, imc.shape[1]))
    imc = imc[ym:yp, xm:xp]
    
    if x is None:
        x = np.arange(imc.shape[1], dtype=float)
    else:
        x = x[xm:xp]
    if y is None:
        y = np.arange(imc.shape[0], dtype=float)
    else:
        y = y[ym:yp]
    
    # Compute beam moments and angle
    cx, cy, rx, ry, theta, gamma = get_ellipse_moments(imc, dx(x), dx(y), cut=0)
    cx += x[0]
    cy += y[0]
    
    # Calculate fluence
    if beam_energy:
        energy_coeff = beam_energy/(np.sum(imc[imc>0])*dx(x)*dx(y))
        imc *= energy_coeff
        
    # Compute flat top beam variations
    top_mean, top_rms, top_ptp = get_flattop_rms(imc, x, y, cx, cy, rx, ry, theta, coeff=0.8)
    
    # Compute tilt of intensity distribution
    if tilt_cutoff is None:
        tilt_cutoff = 0.33*top_mean
    th_tilt, flatness = beam_profile_tilt(imc, tilt_cutoff)
    
    # Fit the whole image with a 2D super-Gaussian
    X, Y = np.meshgrid(x, y)
    try:
        sgfitim = curve_fit_ds(sg2d_fit, X, Y, imc, p0=[rx*2, ry*2, cx, cy, 0,4, top_mean], bounds=([0,0,-np.inf, -np.inf, -np.inf, 0, 0], 7*[np.inf]), Nds=100)
    except RuntimeError:
        print('2D super-Gaussian fit failed, trying lineout fits')
        # Fit X and Y lineouts with super-Gaussian
        try:
            sgfitx = curve_fit(sg_fit, x, imc[np.argmin(np.abs(y-cy)), :], p0=[cx, rx*2, top_mean, 4], bounds=([-np.inf,0,0,0],4*[np.inf]))[0]
        except RuntimeError:
            print('Horizontal super-Gaussian fit failed....')
            sgfitx = [0]*4
        try:
            sgfity = curve_fit(sg_fit, y, imc[:, np.argmin(np.abs(x-cx))], p0=[cy, ry*2, top_mean, 4], bounds=([-np.inf,0,0,0],4*[np.inf]))[0]
        except RuntimeError:
            print('Vertical super-Gaussian fit failed....')
            sgfity = [0]*4
        sgfitim = [None, sgfitx, sgfity]
    
    if sgfitim[0] is not None:
        sg_im_tmp = sg2d_fit((X,Y), *sgfitim)
        avg_tmp, rms_tmp, ptp_tmp = get_flattop_rms(imc-sg_im_tmp+1, x, y, cx, cy, rx, ry, theta, coeff=0.8)
        top_rms = rms_tmp*avg_tmp/top_mean
    
    # Build output dictionary
    output_dict = {'center_x':cx, 'center_y':cy, 'radius_x':rx, 'radius_y':ry,
                    'theta_ell':theta, 'gamma_ell':gamma, 'im_filter':imc,'x':x, 'y':y,
                    'baseline':baseline, 'flatness_axis':th_tilt, 'flatness_data':flatness,
                    'top_mean':top_mean, 'top_rms':top_rms, 'top_ptp':top_ptp,
                    'fit':sgfitim, 'energy':beam_energy, 'unit_en':unit_en}
    
    if threshold_fluence is not None:
        Xp, Yp = np.meshgrid(x-cx, y-cy)
        list_peak = np.array([[xx,yy] for xx, yy, zz in zip(Xp.flatten(), Yp.flatten(), imc.flatten()) if zz>threshold_fluence])
    else:
        list_peak = []
    if threshold_fluence2 is not None:
        Xp, Yp = np.meshgrid(x-cx, y-cy)
        list_peak2 = np.array([[xx,yy] for xx, yy, zz in zip(Xp.flatten(), Yp.flatten(), imc.flatten()) if zz>threshold_fluence2])
    else:
        list_peak2 = []
    if plot:
        beam_analysis_plot(output_dict, x, y, unit=unit, list_peak=list_peak, th_peak=threshold_fluence, list_peak2=list_peak2, th_peak2=threshold_fluence2)
    return output_dict

def get_flattop_rms(image, x, y, cx, cy, rx, ry, theta, coeff=0.8):
    """
    Compute the statistics of the plateau of a flat-top beam.
    
    Parameters
    ----------
    image: 2D np.array
        Image to analyze
        
    x: 1D np.array
        Horizontal axis
    
    y: 1D np.array
        Vertical axis
    
    cx: float
        horizontal position of the centroid
            
    cy: float
        vertical position of the centroid
            
    rx: float
        radius of the beam close to the
        horizontal axis (in the beam reference frame)
            
    ry: float
        radius of the beam close to the
        vertical axis (in the beam reference frame)
            
    theta: float
        angle of the beam major axis 
    
    coeff: float, optional
        fraction of the beam radius to include in the calculation.
        Defaults to 80%.
    
    Output
    ------
    avg: float
        Average value of the plateau
        
    rms: float
        Relative standard deviation of the plateau
        
    ptp: float
        Relative peak-to-peak amplitude of the plateau
    """
    X,Y = np.meshgrid(x-cx,y-cy)
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)
    X_sc = X_rot/(rx*coeff)
    Y_sc = Y_rot/(ry*coeff)
    R = np.sqrt(X_sc**2+Y_sc**2)
    avg = np.nanmean(image[R<=1])
    rms = np.nanstd(image[R<=1])/avg
    ptp = (np.nanmax(image[R<=1]) - np.nanmin(image[R<=1]))/avg
    return avg, rms, ptp
    
def beam_analysis_plot(data, x, y, unit='mm', list_peak=[], th_peak=None, list_peak2=[], th_peak2=None):
    """
    Display image, lineout and beam parameters
    
    Parameters
    ----------
    data: dict
        output dictionary of the function beam_analysis
        
    x: np.array
        horizontal axis
        
    y: np.array
        vertical axis
    
    unit: str, optional
        unit of spatial axes. Default to 'mm'
    
    list_peak: list, optional
        list of pixels above a threshold value. The pixels will be highlighted in orange.
    
    th_peak: float or None, optional
        Value of the threshold used for list_peak.
        
    list_peak2: list, optional
        list of pixels above a threshold value. The pixels will be highlighted in red.
    
    th_peak2: float or None, optional
        Value of the threshold used for list_peak2.
    """
    # Normalize image
    im = data['im_filter']#/data['top_mean']
    # Nice colormap
    cmap_nicify('YlGnBu_r')
    # Center axis
    x -= data['center_x']
    y -= data['center_y']
    
    # Compute axes for the ellipse overlay
    if data['gamma_ell'] < 0:
        thetaplot = data['theta_ell']-np.pi/2
    else:
        thetaplot = data['theta_ell']
    th = np.linspace(0,2*np.pi, 101)
    x_tmp = data['radius_x']*np.cos(th)
    y_tmp = data['radius_y']*np.sin(th)
    x_ell = x_tmp * np.cos(data['theta_ell']) - y_tmp * np.sin(data['theta_ell'])
    y_ell = x_tmp * np.sin(data['theta_ell']) + y_tmp * np.cos(data['theta_ell'])
    iy = np.argmin(np.abs(x))
    ix = np.argmin(np.abs(y))
    maj_ax = np.maximum(data['radius_x'],data['radius_y'])
    # plot limit
    limplot = np.max(np.r_[np.abs(x),np.abs(y)])
    if data['energy']:
        title_unit = data['unit_en'] + '/'+ unit + '^2'
    else:
        title_unit = 'arb. units'
    
    # Figure
    plt.figure(figsize=(4*1.6*2,4), dpi=200)
    # Beam profile plot with ellipse overlay
    plt.subplot(121)
    plt.imshow(im, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect=1, cmap='YlGnBu_r_w', vmin=0, vmax=None)#data['top_mean']*(1+data['top_rms']*5))
    plt.colorbar()
    plt.plot(x_ell, y_ell, c='k', alpha=0.75, ls='--')
    plt.plot(0.8*x_ell, 0.8*y_ell, c='k', alpha=0.75, ls=':')
    plt.plot(np.array([0,maj_ax])*np.cos(thetaplot), np.array([0,maj_ax])*np.sin(thetaplot), c='k', alpha=0.75, ls='--')
    plt.axhline(y[ix], c='crimson', lw=0.5, ls=':')
    plt.axvline(x[iy], c='steelblue', lw=0.5, ls=':')
    if len(list_peak):
        plt.plot(list_peak[:,0], list_peak[:,1], '.', c='darkorange', label = 'F > {:.2f} '.format(th_peak) + title_unit)#, markersize=1)
        plt.legend(loc='upper left', fontsize=8)
    if len(list_peak2):
        plt.plot(list_peak2[:,0], list_peak2[:,1], '.', c='r', label = 'F > {:.2f} '.format(th_peak2) + title_unit)#, markersize=1)
        plt.legend(loc='upper left', fontsize=8)
    plt.xlim(-limplot, limplot)
    plt.ylim(-limplot, limplot)
    plt.xlabel('Horizontal axis ['+unit+']')
    plt.ylabel('Vertical axis ['+unit+']')
    plt.title('Fluence profile ['+title_unit+']')
    
    plt.subplot(122)
    plt.plot(x, im[ix,:], c='crimson', alpha=0.5, label='Horizontal')
    plt.plot(y, im[:,iy], c='steelblue', alpha=0.5, label='Vertical')
    if data['fit'][0] is None:
        plt.plot(x, sg_fit(x, data['fit'][1][0]-data['center_x'], data['fit'][1][1], data['fit'][1][2], data['fit'][1][3]), 
                 c='crimson', label='FWHM = {:.2f} '.format(data['fit'][1][1])+unit+', order = {:.1f}'.format(data['fit'][1][2]))
        plt.plot(y, sg_fit(y, data['fit'][2][0]-data['center_y'], data['fit'][2][1], data['fit'][2][2], data['fit'][2][3]), 
                 c='steelblue', label='FWHM = {:.2f} '.format(data['fit'][2][1])+unit+', order = {:.1f}'.format(data['fit'][2][2]))
    else:
        plt.plot(x, sg_fit(x, data['fit'][2]-data['center_x'], data['fit'][0], data['fit'][5], data['fit'][6]), 
                 c='crimson', label='FWHM = {:.2f} '.format(data['fit'][0])+unit+', order = {:.1f}'.format(data['fit'][5]))
        plt.plot(y, sg_fit(y, data['fit'][3]-data['center_y'], data['fit'][1], data['fit'][5], data['fit'][6]), 
                 c='steelblue', label='FWHM = {:.2f} '.format(data['fit'][1])+unit+', order = {:.1f}'.format(data['fit'][5]))
    plt.xlim(-limplot, limplot)
    plt.ylim(None,plt.ylim()[1]+np.ptp(plt.ylim())/5)
    plt.legend(loc='upper center', ncol=2)
    plt.xlabel('Axis ['+unit+']')
    plt.ylabel('Fluence ['+title_unit+']')
    plt.title('Lineouts')
    
    plt.tight_layout()

    print('Major axis: {:.2f} '.format(2*np.maximum(data['radius_x'],data['radius_y']))+unit)
    print('Minor axis: {:.2f} '.format(2*np.minimum(data['radius_x'],data['radius_y']))+unit)
    print('Angle: {:+.1f}°'.format(np.rad2deg(thetaplot)))
    print('Ellipticity: {:.2f}'.format(np.minimum(data['radius_x'],data['radius_y'])/np.maximum(data['radius_x'],data['radius_y'])))
    if data['fit'][0] is None:
        print('Super-Gaussian order: {:.1f}'.format(0.5*(data['fit'][1][2]+data['fit'][2][2])))
    else:
        print('Super-Gaussian order: {:.1f}'.format(data['fit'][5]))
    print('Flat top RMS: {:.1f} %'.format(data['top_rms']*100))
    print('Flat top PtP: {:.1f} %'.format(data['top_ptp']*100))
    print('Flat top mean fluence: {:.3g} '.format(data['top_mean'])+title_unit)

    
def fourier_transform_spectrum (wavelengths, spectrum, scale_ft_size=1):
    """
    Get the temporal profile from a spectrum (in lambda)
    
    Parameters
    ----------
    wavelengths: 1D np.array
        Wavelength axis
    
    spectrum: 1D np.array
        Intensity spectrum
    
    scale_ft_size: int, optional
        Zero padding for the Fourier transform
    
    Output
    ------
    intensity_profile: 1D np.array,
        Temporal intensity profile
    """
    c = 299792458
    N = wavelengths.size
    N_ft = N * scale_ft_size
    f_direct = c / wavelengths  # frequency axis with unequal spacing
    spectrum_freq = np.sqrt( spectrum * ( c / f_direct**2 ) )  # scaled spectrum in frequency-space
    f = np.linspace( f_direct.min(), f_direct.max(), N ) # frequency axis with equal spacing
    spectrum_freq_interp = interp1d( f_direct, spectrum_freq, kind='cubic' )
    spectrum_freq_interp = spectrum_freq_interp(f) # interpolated spectrum
    intensity_profile = np.abs( np.fft.fftshift( np.fft.ifft( spectrum_freq_interp, n=N_ft ) ) )**2 # Temporal profile
    intensity_profile /= intensity_profile.max()
    t = np.fft.fftfreq( N_ft, f[1] - f[0] ) # time axis
    t = np.r_[ t[N_ft//2:], t[:N_ft//2] ]
    return intensity_profile, t


def beam_profile_tilt(im, bw_cutoff=0.1, Nscan=101):
    """
    Measures the asymmetry/tilt of a flat-top profile
    
    Parameters:
    ===========
    im: numpy.array
        Beam profile to analyze
        
    bw_cutoff: float, optional
        Threshold to identify the geometric center
        
    Nscan: integer, optional
        Resolution of the angular scan
    """
    im = im.T
    # Define axes
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    X, Y = np.meshgrid(x,y)
    # Get geometric center
    mask = np.zeros_like(im)
    mask[im>=bw_cutoff] = 1
    gx, gy, _, _ = get_moments(mask)
    # Define rotating variables
    R,T = cart2pol(X-gx,Y-gy)
    theta = np.linspace(-1,1,Nscan)*np.pi
    flatness = np.zeros(Nscan)
    # Scan the angle
    for i, th in enumerate(theta):
        flatness[i] = np.sum(im[np.mod(T+th, 2*np.pi)<np.pi])
    flatness /= np.sum(im)/2
    return theta, flatness
    
def sg_fit(x, x0, fwhm, order, amplitude, offset=0):
    """"
    Super-Gaussian formula to use with curve_fit
    
    Parameters
    ----------
    x: float 1D np.array
        Axis of the Gaussian

    x0: float
        Center position of the Gaussian
    
    fwhm: float
        Full Width at Half Maximum
        
    order: float
        order of the super-Gaussian function.
        Defined as: exp( - ( x**2 )**order )

    amplitude: float
        Amplitude of super-Gaussian
        
    offset: float, optional
        Amplitude offset of the Gaussian
    """
    return amplitude*np.exp(-np.log(2) * ((2 * (x - x0) / fwhm)**2)**(order/2)) + offset

def sg2d_fit(x, fwx, fwy, x0, y0, offset, order, amplitude):
    """
    2D Super-Gaussian formula to used with curve_fit
    """
    X, Y = x[0], x[1]
    return amplitude*gauss2D(X, Y, fwx, fwy, x0, y0, offset, order/2, int_FWHM=False)

def downscale_mean(arr, red):
    """
    Downscale an image by the local mean
    
    Parameters
    ----------
    arr: 2D numpy.array
        Array to reduce
        
    red: int, or couple
        Factor by how much the array is reduced.
        If couple, the first factor reduces in x, and the second in y.
    
    Output
    ------
    arr_out: 2D numpy.array
        Reduced array
    
    """
    if isinstance(red, (int, float)):
        redx, redy = red, red
    else:
        redx, redy = red[0], red[1]
    ny, nx = arr.shape
    arr_out = np.zeros([ny//redy, nx//redx])
    iy, ix = np.unravel_index(np.arange(redx*redy), (redy,redx))
    for k in range(redx*redy):
        arr_out += arr[iy[k]:ny//redy*redy:redy, ix[k]:nx//redx*redx:redx]
    return arr_out/(redx*redy)
    
def curve_fit_ds(f, x, y, image, p0=None, bounds=(-np.inf, np.inf), Nds=100):
    """
    Image curve fitting, with downsampling to reduce computation time
    """
    ny, nx = image.shape
    dsx = nx//Nds
    dsy = ny//Nds
    im_red = downscale_mean(image, (dsy, dsx))
    x_red = downscale_mean(x, (dsy, dsx))
    y_red = downscale_mean(y, (dsy, dsx))
    return curve_fit(f, (x_red.flatten(), y_red.flatten()), im_red.flatten(), p0=p0, bounds=bounds)[0]
    