import numpy as np
import matplotlib.pyplot as plt
import warnings

class Beampath:
    """
    Computes the beam radius and angle according to ABCD matrices using paraxial approximation. Units should be consistent (all mm or all m)
    """
    # Define plotting variables
    __col_beam = 'crimson'
    __col_object = 'tab:blue'
    __col_lens_pos = 'tab:green'
    __col_lens_neg = 'tab:orange'
    __col_image = 'purple'
    __col_exit = 'darkslategrey'
    __col_inter = 'xkcd:dull blue'
    __alpha_beam = 0.1
    __scale_vert = 0.8
    __text_offset = 0.05
    __xmin = 0
    __xmax = 1
    __ymin = -1
    __ymax = 1
    __prec = '%.2f'
    
    def __init__(self, radius=0, angle=0.1, index=1, position=0):
        """
        Initialise beam path with an object
        
        Parameters
        ----------
        radius: float, optional
            Radius of the object
            
        angle: float, optional
            Half divergence angle in radians
            
        index: float, optional
            Initial index of refraction
            
        position: float, optional
            Longitudinal position of the object
        """
        self.radius = np.array([radius])
        self.angle = np.array([angle])
        if index < 1:
            warnings.warn('Index lower than 1!')
        self.index = np.array([index])
        self.position = np.array([position])
        self.list_elements = [Object()]
        self.M = np.identity(2)
        self._check_paraxial()
    
    def edit_object(self, radius=None, angle=None, position=None):
        """
        Change object and recalculate beam path
        
        Parameters
        ----------
        radius: float
            Object radius
        
        angle: float
            Object angle in radians
        
        position: float
            Object position
        """
        if radius is not None:
            self.radius[0] = radius
        if angle is not None:
            self.angle[0] = angle
        if position is not None:
            self.position[0] = position
        self._recalculate()
    
    def add_thinlens(self, focal_length):
        """
        Add a thin lens to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        focal_length: float
            Focal length of the thin lens.
            Can be positive or negative but not zero
        """
        self.list_elements.append(Thinlens(focal_length))
        new_rad, new_angle = np.dot(self.list_elements[-1].M, np.array([self.radius[-1], self.angle[-1]]))
        self.radius = np.append(self.radius, new_rad)
        self.angle = np.append(self.angle, new_angle)
        self.index = np.append(self.index, self.index[-1])
        self.position = np.append(self.position, self.position[-1])
        self.M = np.dot(self.list_elements[-1].M, self.M)
        self._check_paraxial()
    
    def add_freespace(self, distance):
        """
        Add free space propagation to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        distance: float
            Distance of propagation in free space.
            Can be negative but be careful of the results.
            Cannot be zero.
        """
        self.list_elements.append(Freespace(distance))
        new_rad, new_angle = np.dot(self.list_elements[-1].M, np.array([self.radius[-1], self.angle[-1]]))
        self.radius = np.append(self.radius, new_rad)
        self.angle = np.append(self.angle, new_angle)
        self.index = np.append(self.index, self.index[-1])
        self.position = np.append(self.position, self.position[-1]+distance)
        self.M = np.dot(self.list_elements[-1].M, self.M)
        self._check_paraxial()
    
    def add_interface(self, index, curvature = np.inf):
        """
        Add an interface with a different medium to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        index: float
            Index of refraction after the interface.
            
        curvature: float, optional
            Radius of curvature of the interface.
            Positive means convexe surface, negative means concave surface.
            Infinite radius is a flat surface.
        """
        self.list_elements.append(Interface(index, self.index[-1], curvature))
        new_rad, new_angle = np.dot(self.list_elements[-1].M, np.array([self.radius[-1], self.angle[-1]]))
        self.radius = np.append(self.radius, new_rad)
        self.angle = np.append(self.angle, new_angle)
        self.index = np.append(self.index, index)
        self.position = np.append(self.position, self.position[-1])
        self.M = np.dot(self.list_elements[-1].M, self.M)
        self._check_paraxial()
    
    def add_thicklens(self, index, thickness, radius_in=np.inf, radius_out=np.inf):
        """
        Add a thick lens to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        index: float
            Index of refraction of the thick lens.
            
        thickness: float
            Center thickness of the thick lens.
            
        radius_in: float, optional
            Radius of curvature of the first surface.
            Positive means convexe surface, negative means concave surface.
            Infinite radius is a flat surface.
        
        radius_out: float, optional
            Radius of curvature of the second surface.
            Positive means convexe surface, negative means concave surface.
            Infinite radius is a flat surface.
        """
        index_before = self.index[-1]
        self.add_interface(index, radius_in)
        self.add_freespace(thickness)
        self.add_interface(index_before, radius_out)
    
    def add_image(self):
        """
        Calculate the image position and add it to the beam path
        """
        beam_in = np.array([0,1])
        beam_out = np.dot(self.M, beam_in)
        # Get distance where the beam crosses the optical axis
        with np.errstate(divide='ignore'): # remove divide by zero warning
            distance_image = -beam_out[0]/beam_out[1]
        if isinstance(self.list_elements[-1], Freespace):
            self._add_exit()
        self.add_freespace(distance_image)
        self.list_elements.append(Image())
        self.radius = np.append(self.radius, self.radius[-1])
        self.angle = np.append(self.angle, self.angle[-1])
        self.index = np.append(self.index, self.index[-1])
        self.position = np.append(self.position, self.position[-1])
        self._check_paraxial()
    
    def remove_element(self, element_index):
        """
        Remove the surface placed in position element_index
        
        Parameter
        ---------
        element_index: int or list or int
            Index of the element to remove.
            Should be between 1 and the last element
        """
        # If input is integer, put it in a list
        try:
            element_index[0]
        except:
            element_index = [element_index]
            
        # Check range of input
        if min(element_index) < 1:
            raise ValueError("Impossible to remove the object...")
        elif max(element_index) > len(self.list_elements):
            raise ValueError("Index too high...")
                
        # Check if last element is an image
        add_image = False
        if isinstance(self.list_elements[-1], Image):
            if isinstance(self.list_elements[-3], Exit):
                element_index.append(len(self.list_elements)-3)
            element_index.append(len(self.list_elements)-2)
            element_index.append(len(self.list_elements)-1)
            add_image = True
        # Get the new list and recalculate
        new_list = [elem for i, elem in enumerate(self.list_elements) if i not in element_index]
        self.list_elements = new_list
        self._recalculate()
        if add_image:
            self.add_image()
    
    def pop(self):
        """
        Remove the last element (apart from the image) and returns it
        """
        if isinstance(self.list_elements[-1], Image):
            if isinstance(self.list_elements[-3], Exit):
                last_element = self.list_elements[-4]
                idx = -4
            else:
                last_element = self.list_elements[-3]
                idx = -3
        else:
            last_element = self.list_elements[-1]
            idx = -1
        
        self.remove_element(len(self.list_elements)+idx)
        return last_element
    
    def extend(self, beampath):
        """
        Append a beampath to the current one
        
        Parameters
        ----------
        beampath: Beampath
            beampath to append
        """
        # Check if new beampath has an image
        if isinstance(beampath.list_elements[-1], Image):
            if isinstance(beampath.list_elements[-3], Exit):
                lastidx = -4
            else:
                lastidx = -3
        else:
            lastidx = -1
        lastidx += len(beampath.list_elements)+1
        list_to_append = beampath.list_elements[1:lastidx]
        # Check if self has an image
        add_image = False
        if isinstance(self.list_elements[-1], Image):
            self.remove_image()
            add_image = True
        self.list_elements.extend(list_to_append)
        self._recalculate()
        if add_image:
            self.add_image()
        
    def remove_image(self):
        """
        Remove image from the beam path
        """
        if isinstance(self.list_elements[-1], Image):
            if isinstance(self.list_elements[-3], Exit):
                idx_remove = -4
            else:
                idx_remove = -3
            self.list_elements = self.list_elements[:(idx_remove+1)]
            self._recalculate()
            
    
    def _add_exit(self):
        """
        Add an dummy exit plane
        """
        self.list_elements.append(Exit())
        self.radius = np.append(self.radius, self.radius[-1])
        self.angle = np.append(self.angle, self.angle[-1])
        self.index = np.append(self.index, self.index[-1])
        self.position = np.append(self.position, self.position[-1])
    
    def _check_paraxial(self):
        """
        Check if the paraxial approximation is respected, i.e. theta-sin(theta) is small enough
        """
        par_error = np.max(np.abs(self.angle - np.sin(self.angle)))
        if par_error > 0.01:
            warn_message = 'Paraxial approximation is not valid: |x - sin(x)| = %.3g'%par_error
            warnings.warn(warn_message)
    
    def _recalculate(self):
        """
        Recalculate the beampath parameters
        """
        # Reset the parameters
        self.radius = np.array([self.radius[0]])
        self.angle = np.array([self.angle[0]])
        self.index = np.array([self.index[0]])
        self.position = np.array([self.position[0]])
        self.M = np.identity(2)
        
        # Go through the list of components and recalculate
        for idx, element in enumerate(self.list_elements):
            if not isinstance(element, Object):
                if isinstance(element, Interface):
                    element.index_in = self.index[-1]
                    element.compute_matrix()
                    new_index = element.index_out
                else:
                    new_index = self.index[-1]
                new_position = self.position[-1]
                if isinstance(element, Freespace):
                    new_position += element.distance
                new_rad, new_angle = np.dot(self.list_elements[idx].M, np.array([self.radius[-1], self.angle[-1]]))
                self.radius = np.append(self.radius, new_rad)
                self.angle = np.append(self.angle, new_angle)
                self.index = np.append(self.index, new_index)
                self.position = np.append(self.position, new_position)
                self.M = np.dot(self.list_elements[idx].M, self.M)
        self._check_paraxial()


    def plot(self, **kwargs):
        """
        Plot the beam path
        
        Optionnal keywords arguments
        ----------------------------
        'figsize': int tuple
            Set the figure size
        
        'plot_digit': int
            Change the maximum number of decimal digits of the annotations
        """
        fig, ax = self._plot_initialise(**kwargs)
        for idx in range(len(self.list_elements)):
            self.list_elements[idx].plot(ax, idx, self, **kwargs)
        
    
    def _plot_initialise(self, **kwargs):
        """
        Initialise plot and plot variables
        """
        fig_size = kwargs.get('figsize', (12,6))
        fig, ax = plt.subplots(figsize=fig_size) # Initialise figure
        pos = self.position[np.isfinite(self.position)]
        xrange = np.ptp(pos) # Get x range
        # Compute x and y limits
        if self.list_elements:
            self.__xmin = np.min(pos) - 0.1*xrange
            self.__xmax = np.max(pos) + 0.1*xrange
            self.__ymax = np.max(np.abs(self.radius[np.isfinite(self.radius)]))*1.5
            self.__ymin = - self.__ymax
        # Set x and y limits
        ax.set_xlim(self.__xmin, self.__xmax)
        ax.set_ylim(self.__ymin, self.__ymax)
        # Define the digit precision
        plot_digit = kwargs.get('plot_digit', None)
        if plot_digit is not None:
            self.__prec = '%.' + '%d'%plot_digit + 'f'
        # Plot optical axis
        ax.plot([self.__xmin, self.__xmax],[0, 0], ls='-.', color='grey')
        if np.any(self.index!=self.index[0]):
            tx_off = self._Beampath__text_offset
            sc_vert = self._Beampath__scale_vert
            col = self._Beampath__col_inter
            ax.text(self.__xmin + 0.01*xrange, self.__ymin*(tx_off+sc_vert), 
                    'Index:', color = col, ha='left', va='top')
        return fig, ax
    
class GaussianBeampath(Beampath):
    """
    Computes the beam radius and angle according to ABCD matrices using paraxial approximation and the complex Gaussina parameter. Units should be consistent (all mm or all m)
    """
    
    def __init__(self, radius=0, angle=0.1, index=1, position=0, wavelength=8e-7):
        """
        Initialise beam path with an object
        
        Parameters
        ----------
        radius: float, optional
            Radius of the object
            
        angle: float, optional
            Half divergence angle in radians
            
        index: float, optional
            Initial index of refraction
            
        position: float, optional
            Longitudinal position of the object
        """
        super().__init__(radius, angle, index, position)
        self.wavelength = wavelength
        
   
    def add_thinlens(self, focal_length):
        """
        Add a thin lens to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        focal_length: float
            Focal length of the thin lens.
            Can be positive or negative but not zero
        """
        self.list_elements.append(Thinlens(focal_length))
        q1 = rth_to_q(self.radius[-1], self.angle[-1], self.index[-1], self.wavelength)
        q2 = gauss_abcd(q1, self.list_elements[-1].M)
        new_rad, new_angle = q_to_rth(q2, self.index[-1], self.wavelength)
        self.radius = np.append(self.radius, new_rad)
        self.angle = np.append(self.angle, new_angle)
        self.index = np.append(self.index, self.index[-1])
        self.position = np.append(self.position, self.position[-1])
        self.M = np.dot(self.list_elements[-1].M, self.M)
        self._check_paraxial()
    
    def add_freespace(self, distance):
        """
        Add free space propagation to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        distance: float
            Distance of propagation in free space.
            Can be negative but be careful of the results.
            Cannot be zero.
        """
        self.list_elements.append(Freespace(distance))
        q1 = rth_to_q(self.radius[-1], self.angle[-1], self.index[-1], self.wavelength)
        q2 = gauss_abcd(q1, self.list_elements[-1].M)
        new_rad, new_angle = q_to_rth(q2, self.index[-1], self.wavelength)
        self.radius = np.append(self.radius, new_rad)
        self.angle = np.append(self.angle, new_angle)
        self.index = np.append(self.index, self.index[-1])
        self.position = np.append(self.position, self.position[-1]+distance)
        self.M = np.dot(self.list_elements[-1].M, self.M)
        self._check_paraxial()
    
    def add_interface(self, index, curvature = np.inf):
        """
        Add an interface with a different medium to the optical path and calculates the new beam parameters
        
        Parameters
        ----------
        index: float
            Index of refraction after the interface.
            
        curvature: float, optional
            Radius of curvature of the interface.
            Positive means convexe surface, negative means concave surface.
            Infinite radius is a flat surface.
        """
        self.list_elements.append(Interface(index, self.index[-1], curvature))
        self.index = np.append(self.index, index)
        q1 = rth_to_q(self.radius[-1], self.angle[-1], self.index[-1], self.wavelength)
        q2 = gauss_abcd(q1, self.list_elements[-1].M)
        new_rad, new_angle = q_to_rth(q2, self.index[-1], self.wavelength)
        self.radius = np.append(self.radius, new_rad*np.sqrt(self.index[-2]/self.index[-1]))
        self.angle = np.append(self.angle, new_angle)
        
        self.position = np.append(self.position, self.position[-1])
        self.M = np.dot(self.list_elements[-1].M, self.M)
        self._check_paraxial()

    
class Element:
    """
    Abstract parent class for optical elements.
    Do not call!
    """
    def __init__(self):
        # Initialise plotting variables
        self.__color = {'beam': 'crimson',
                        'object': 'tab:blue',
                        'lens_pos': 'tab:green',
                        'lens_neg': 'tab:orange',
                        'image': 'purple',
                        'exit': 'darkslategrey',
                        'interface': 'xkcd:dull blue',
                        'axis': 'grey'}
        self.__alpha_beam = 0.1
        self.__scale_vert = 0.8
        self.__text_offset = 0.05
        self.__xmin = 0
        self.__xmax = 1
        self.__ymin = -1
        self.__ymax = 1
        self.__prec = '%.2f'
    
    def _plot_var(self, ax, **kwargs):
        """
        Initialise the plot variables
        """
        self.__xmin, self.__xmax = ax.get_xlim()
        self.__ymin, self.__ymax = ax.get_ylim()
        plot_digit = kwargs.get('plot_digit', None)
        if plot_digit is not None:
            self.__prec = '%.' + '%d'%plot_digit + 'f'
    
    def compute_matrix(self):
        self.M = np.identity(2)
        
        
class Object(Element):
    """
    Object of beam path
    """
    def __init__(self):
        super().__init__()
        # Initialise matrix
        self.compute_matrix()
    
    def plot(self, ax, elem_pos, beam_path, **kwargs):
        """
        Plot the object in the beam path
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Axis in which the object should be drawn
        
        elem_pos: float
            Longitudinal position where the object should be drawn
        
        beam_path: Beampath
            Beampath containing the object
        """
        # Get plot variables
        self._plot_var(ax, **kwargs)
        ymin = self._Element__ymin
        ymax = self._Element__ymax
        sc = self._Element__scale_vert
        tx = self._Element__text_offset
        col = self._Element__color['object']
        pos = beam_path.position[elem_pos]
        # Plot object
        ax.plot([pos, pos], [ymin*sc, ymax*sc], ls='--', color = col)
        ax.text(pos, ymax*(sc+tx), 'Object', color = col, ha = 'center', va = 'bottom')
    
class Thinlens(Element):
    """
    Thin lens optical element
    """
    def __init__(self, focal_length):
        """
        Constructor of the thin lens element
        
        Parameters
        ----------
        focal_length: float
            Focal length of the thin lens
        """
        super().__init__()
        self.focal_length = focal_length
        self.compute_matrix()
    
    def compute_matrix(self):
        self.M = np.array([[1,0],[-1/self.focal_length, 1]])
    
    def plot(self, ax, elem_pos, beam_path, **kwargs):
        """
        Plot the thin lens in the beam path
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Axis in which the thin lens should be drawn
        
        elem_pos: float
            Longitudinal position where the thin lens should be drawn
        
        beam_path: Beampath
            Beampath containing the thin lens
        """
        # Get plot variables
        self._plot_var(ax, **kwargs)
        ymin = self._Element__ymin
        ymax = self._Element__ymax
        sc = self._Element__scale_vert
        tx = self._Element__text_offset
        prec = self._Element__prec
        prec = prec[0] + '+' + prec[1:] # display signed focal length
        pos = beam_path.position[elem_pos]
        if self.focal_length < 0:
            col = self._Element__color['lens_neg']
        else:
            col = self._Element__color['lens_pos']
        # Plot lens with text
        ax.plot([pos, pos], [sc*ymin, sc*ymax], color=col)
        ax.text(pos, (sc+tx)*ymax, ('Lens\n'+prec%(self.focal_length)).rstrip('0').rstrip('.'),
                color=col, ha='center', va='bottom')
        

class Freespace(Element):
    """
    Free space propagation
    """
    def __init__(self, distance):
        """
        Constructor of the thin lens element
        
        Parameters
        ----------
        distance: float
            Distance of the free space propagation
        """
        super().__init__()
        self.distance = distance
        self.compute_matrix()
    
    def compute_matrix(self):
        self.M = np.array([[1,self.distance],[0, 1]])
    
    def plot(self, ax, elem_pos, beam_path, **kwargs):
        """
        Plot the propagated beam path
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Axis in which the thin lens should be drawn
        
        elem_pos: float
            Index of Freespace in beam path
        
        beam_path: Beampath
            Beampath containing the free space
        """
        # Get plot variables
        self._plot_var(ax, **kwargs)
        ymin = self._Element__ymin
        ymax = self._Element__ymax
        sc = self._Element__scale_vert
        tx = self._Element__text_offset
        col = self._Element__color['beam']
        col2 = self._Element__color['interface']
        alp = self._Element__alpha_beam
        if isinstance(beam_path, GaussianBeampath):
            tmp_bp = GaussianBeampath(radius=beam_path.radius[elem_pos-1],
                                angle=beam_path.angle[elem_pos-1],
                                index=beam_path.index[elem_pos-1],
                                wavelength=beam_path.wavelength)
            N = 100
            step = self.distance/N
            for i in range(N):
                tmp_bp.add_freespace(step)
            rad = tmp_bp.radius
            pos = tmp_bp.position
            pos += beam_path.position[elem_pos-1]
        else:
            pos = beam_path.position[elem_pos-1:elem_pos+1]
            rad = beam_path.radius[elem_pos-1:elem_pos+1]
        pos_lim = beam_path.position[elem_pos-1:elem_pos+1]
        n = beam_path.index[elem_pos]
        prec = self._Element__prec
        # Plot beam path
        # If beam going to image, plot the beam in dotted line otherwise, solid line
        if isinstance(beam_path.list_elements[-1], Image) and elem_pos == (len(beam_path.list_elements)-2):
            line_style = ':'
        else:
            line_style = '-'
        # Beam edges
        ax.plot(pos, rad,line_style, color=col)
        ax.plot(pos, -rad,line_style, color=col)
        # Beam inside
        ax.fill_between(pos, rad, -rad, facecolor=col, alpha=alp)
        # If there is a change of index, overlay a shadowed area according to the refractive index
        if np.any(beam_path.index!=beam_path.index[0]):
            ax.fill_between(pos_lim, [ymax, ymax], [ymin, ymin], facecolor = col2, alpha = (1-1/n))
            ax.text(np.mean(pos_lim), (sc+tx)*ymin, (prec%n).rstrip('0').rstrip('.'), 
                        color=col2, ha = 'center', va='top')

class Interface(Element):
    """
    Interface between two mediums
    """
    def __init__(self, index_out, index_in = 1, curvature=np.inf):
        """
        Constructor of the interface element
        
        Parameters
        ----------
        index_out: float
            Index of refraction after the interface.
        
        index_in: float, optional
            Index of refraction before the interface.
        
        curvature: float, optional
            Radius of curvature of the interface.
            Positive is convexe, negative is concave.
            Infinity is a flat interface.
        """
        super().__init__()
        if index_in < 1 or index_out < 1:
            warnings.warn('Index lower than 1!')
        self.index_in = index_in
        self.index_out = index_out
        self.curvature = curvature
        self.compute_matrix()
    
    def compute_matrix(self):
        self.M = np.array([[1,0],[(self.index_in-self.index_out)/(self.curvature*self.index_out),
                                  self.index_in/self.index_out]])
    
    def plot(self, ax, elem_pos, beam_path, **kwargs):
        """
        Plot the interface
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Axis in which the interface should be drawn
        
        elem_pos: float
            Longitudinal position where the interface is placed
        
        beam_path: Beampath
            Beampath containing the interface
        """
        # Retreive plot variables
        self._plot_var(ax, **kwargs)
        ymin = self._Element__ymin
        ymax = self._Element__ymax
        pos = beam_path.position[elem_pos]
        diff_pos = np.ptp(beam_path.position)/20
        col = self._Element__color['interface']
        # If infinite radius, draw a vertical line
        if not np.isfinite(self.curvature):
            x = np.array([pos,pos])
            y = np.array([0, ymax])
        else: # Otherwise, draw a curved interface
            r = self.curvature
            x0 = r + pos
            x = np.linspace(-r, -r + np.sign(r)*diff_pos, num=200) + x0
            y = np.abs(r*np.sin(np.arccos(np.clip((x-x0)/r, -1,1))))
        ax.plot(x, y, color = col)
        ax.plot(x, -y, color = col)

        
class Exit(Element):
    """
    Exit plane of beam path
    """
    
    def __init__(self):
        super().__init__()
        # Initialise matrix
        self.compute_matrix()
        
    def plot(self, ax, elem_pos, beam_path, **kwargs):
        """
        Plot the exit plane
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Axis in which the exit plane should be drawn
        
        elem_pos: float
            Longitudinal position where the exit plane is placed
        
        beam_path: Beampath
            Beampath containing the exit plane
        """
        # Get plot variables
        self._plot_var(ax, **kwargs)
        ymin = self._Element__ymin
        ymax = self._Element__ymax
        sc = self._Element__scale_vert
        tx = self._Element__text_offset
        col = self._Element__color['exit']
        prec = self._Element__prec
        pos = beam_path.position[elem_pos]        
        # Plot object
        ax.plot([pos, pos], [ymin*sc, ymax*sc], ls='--', color = col)
        #ax.text(pos, ymax*(sc+tx), 'Exit plane at '+prec%pos, color = col, ha = 'center', va = 'bottom')
        
class Image(Element):
    """
    Image of beam path
    """
    
    def __init__(self):
        super().__init__()
        # Initialise matrix
        self.compute_matrix()
    
    def plot(self, ax, elem_pos, beam_path, **kwargs):
        """
        Plot the image plane
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
            Axis in which the image plane should be drawn
        
        elem_pos: float
            Longitudinal position where the image plane is placed
        
        beam_path: Beampath
            Beampath containing the image plane
        """
        # Get plot variables
        self._plot_var(ax, **kwargs)
        xmin = self._Element__xmin
        xmax = self._Element__xmax
        ymin = self._Element__ymin
        ymax = self._Element__ymax
        sc = self._Element__scale_vert
        tx = self._Element__text_offset
        col = self._Element__color['image']
        prec = self._Element__prec
        pos = beam_path.position[elem_pos]
        xrange = xmax-xmin
        yrange = ymax-ymin
        # Plot object
        # If at finite distance, draw the plane
        if np.isfinite(pos):
            ax.plot([pos, pos], [ymin*sc, ymax*sc], ls='--', color=col)
            ax.text(pos, ymax*(sc+tx), ('Image at\n'+prec%pos).rstrip('0').rstrip('.'), color=col, ha='center', va='bottom')
        else: # Othewise, annote the figure that the image is at infinity
            ax.text(xmax - 0.01*xrange, yrange*0.01, 'Image', color=col, ha='right', va='bottom')
            ax.text(xmax - 0.01*xrange, -yrange*0.01, 'at \u221E \u2192', color=col, ha='right', va='top')

def gauss_abcd(q1, M):
    A = M[0,0]
    B = M[0,1]
    C = M[1,0]
    D = M[1,1]
    q2 = (A*q1+B)/(C*q1+D)
    return q2

def rth_to_q(radius, angle, index=1.0, wavelength=8e-7):
    curv = radius/np.tan(angle)
    qinv = 1/curv - 1j*wavelength/(np.pi*index*radius**2)
    return 1/qinv

def q_to_rth(q, index=1.0, wavelength=8e-7):
    qinv = 1/q
    curv = 1/qinv.real
    radius = np.sqrt(-(wavelength/qinv.imag)/np.pi/index)
    angle = np.arctan(radius/curv)
    return radius, angle

def waist_from_nf(radius, angle, wavelength):
    """
    Calculates the Gaussian beam waist parameters from a near field radius and divergence
    """    
    w0 = radius * np.sqrt( ( 1 - np.sqrt( 1 - ( 2*wavelength / ( radius * np.pi * np.tan(angle) ) )**2 ) ) / 2 )
    zr = w0**2*np.pi/wavelength
    z0 = -radius / np.tan(angle)
    return w0, zr, z0

