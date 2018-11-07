"""
A simple library with classes representing various voids in a BCC 
(body centered cubic) structure.
"""
from scipy.spatial import ConvexHull
import numpy as np
import os
import copy
#
from VASP_common_utilites import *
############################################################################################################################
#
#               Helpers
#
############################################################################################################################
def rotate_void_disp_map (direction_, angle, Nxyz, cell_disp, void, origin_shift, edge_shift):
    """
    A service function: it returns the PBC-ced void inside the SC rotated and scaled
    """
    # the rotational matrix
    R = arbitrary_dir_rotation(direction_,angle)
    # shift the origin
    dT_ = void - origin_shift
    # the rotated void + shift the cell and the edge
    RotVoid = np.dot(R,dT_.T).T + origin_shift + edge_shift + cell_disp
    # scale it
    void_scaled  = np.zeros(RotVoid.shape)
    for i in range(Nxyz.shape[0]):
        void_scaled[:,i] =  RotVoid[:,i] / Nxyz[i]
    # implement PBC:
    void_scaled_PBC, PBC_map = implement_PBC(void_scaled,flag_pbc_map=True)
    return void_scaled_PBC, PBC_map

def get_volumes (initial_positions, new_positions, new_lattice, N_XYZ,cell_displacements,class_id, degeneracy):
    """
    Calculates the volumes of voids specified by class_id
    """
    VOIDS = {}
    void_num = 0
    # initialize the objects
    for d_idx in range(cell_displacements.shape[0]):
        cell_disp = cell_displacements[d_idx,:]
        VOIDS[d_idx] = class_id(cell_disp,N_XYZ, initial_positions)
        void_num += VOIDS[d_idx].number_voids
    volumes = np.zeros(void_num)
    cnt = 0
    # identify the volumes in the relaxed structure
    for key in VOIDS.keys():
        try:
            v_d = VOIDS[key].extract_void_coordinates(new_positions,new_lattice)
            for key_v in v_d.keys():
                hull = ConvexHull(v_d[key_v])
                volumes[cnt]=hull.volume
                cnt +=1
        except:
            print('Shit happened at cell %d' % key)
            pass
    # due to methods of initialization of the voids, "volumes"
    # contains pairs of the same volume (within precision, of course)
    # since each void is counted several times, therefore such 
    # multiplets should be removed
    return sparse_array_for_degeneracy(volumes,degeneracy_deg=degeneracy) 

def scale_void_get_PBCmap (_void_, _cell_disp_, _NXYZ_):
    """
    Simply scales the input _void_ according to 
    the super cell (SC) dimensions given in _NXYZ_.
    The function also displaces the void to a 
    unit cell (UC) inside the SC defined in 
    _cell_disp_ vector.
    """
    AVoid = _void_ + _cell_disp_
    void_scaled  = np.zeros(AVoid.shape)
    for i in range(_NXYZ_.shape[0]):
        void_scaled[:,i] =  AVoid[:,i] / _NXYZ_[i]
    void_scaled_PBC, PBC_map = implement_PBC(void_scaled,flag_pbc_map=True)
    return void_scaled_PBC, PBC_map
############################################################################################################################
#
#               Void classes
#
############################################################################################################################
class TetraVoid():      
    def __init__(self,cell_disp, Nxyz, init_pos):
        TetraVoids = {}
        # a seed of voids in XY plane
        TetraVoids['XY'] = np.array([[0,0,0],
                         [0,1,0],
                         [0.5,0.5,0.5],
                         [0.5,0.5,-0.5]])
        # a seed of voids in ZX plane
        TetraVoids['XZ'] = np.array([[0,0,0],
                                   [1,0,0],
                                   [0.5,0.5,0.5],
                                   [0.5,-0.5,0.5]])
        # a seed of voids in YZ plane
        TetraVoids['YZ'] = np.array([[0,0,0],
                                   [0,0,1],
                                   [0.5,0.5,0.5],
                                   [-0.5,0.5,0.5]])
        # angles to rotate for:
        angles = np.array([0,-90,-180,-270])
        # directions to rotate around:
        directions = {}
        directions['XY'] = np.array([0,0,1])
        directions['XZ'] = np.array([0,1,0])
        directions['YZ'] = np.array([1,0,0])
        # displacement vectors for for the edges:
        # bottom --> top; left --> right; front --> back
        edge_displacements = {}
        edge_displacements['XY'] = np.array([[0,0,0],
                                            [0,0,1]])
        edge_displacements['XZ'] = np.array([[0,0,0],
                                            [0,1,0]])
        edge_displacements['YZ'] = np.array([[0,0,0],
                                            [1,0,0]])
        # origin shifts:
        origin_shifts = {}
        origin_shifts['XY'] = np.array([0.5,0.5,0.5])
        origin_shifts['YZ'] = np.array([0.5,0.5,0.5])
        origin_shifts['XZ'] = np.array([0.5,0.5,0.5])
        CNT = 0
        self.voids = {}
        for plane in TetraVoids.keys():
            direction = directions[plane]
            origin_shift = origin_shifts[plane]
            for ang_idx in range(angles.shape[0]):
                edge_disp = edge_displacements[plane]
                for edge_disp_idx in range(edge_disp.shape[0]):
                    angle = angles[ang_idx]
                    edge_d = edge_disp[edge_disp_idx,:]
                    self.voids[CNT]={}
                    void = TetraVoids[plane]
                    TetraVoidSC_PBC, PBC_map= rotate_void_disp_map(direction, angle, Nxyz,
                                                                   cell_disp,void,origin_shift,edge_d)
                    self.voids[CNT]['TetraVoidSC_PBC'] = np.copy(TetraVoidSC_PBC)
                    self.voids[CNT]['vertex_disp'] = np.copy(PBC_map)
                    self.voids[CNT]['plane'] = plane
                    self.voids[CNT]['edge_shift'] = edge_d
                    # find indeces in the original dataset:
                    self.voids[CNT]['indeces'] = find_indeces(init_pos,TetraVoidSC_PBC)
                    CNT +=1
        self.number_voids = CNT
        self.cell_position = np.copy(cell_disp)
        self._init_pos = np.copy(init_pos)
        
    def extract_void_coordinates(self, new_positions, lattice):
        voids = {}
        coords2unwrap = np.append(self._init_pos,new_positions,axis=0)
        unw = unwrap_PBC(coords2unwrap,self._init_pos.shape[0])
        new_coords_unw = unw[self._init_pos.shape[0]:,:]
        for key in self.voids.keys():
            direct_coord_void = self.voids[key]['TetraVoidSC_PBC']
            disps = self.voids[key]['vertex_disp']
            indeces = self.voids[key]['indeces']
            void_direct_coords = new_coords_unw[indeces,:].reshape(direct_coord_void.shape[0],3) + disps 
            voids[key] = frac2cart(void_direct_coords,lattice)
        return voids

class OctaVoidType1():
    def __init__(self,cell_disp, Nxyz, init_pos):
        OctaVoids = {}
        OctaVoidSeed = np.array([[0,0,0],
                                 [0,0,1],
                                 [0.5,0.5,0.5],
                                 [-0.5,0.5,0.5],
                                 [-0.5,-0.5,0.5],
                                 [0.5,-0.5,0.5],
                                ])
        # each void can be created from the seed using this displacements
        displacements = np.array([[0,0,0],
                                 [0,1,0],
                                 [1,1,0],
                                 [1,0,0]
                                 ])
        # they will be rotated aroung these directions
        # in order to cover all possible locations
        # of the octahedral self.voids
        direction2rotate_around = np.array([[0,1,0],
                                           [0,0,1],
                                           [1,0,0]
                                           ])
        # this is the angle to rotate for around each axis
        # specified above
        angle = -90
        cnt = 0
        for dir_idx in range(direction2rotate_around.shape[0]):
            direction = direction2rotate_around[dir_idx]
            R = arbitrary_dir_rotation(direction,angle)
            for disp_idx in range(displacements.shape[0]):
                # all the self.voids belonging to each type and based on
                # specific edge/plane can be created from 
                OctaVoids[cnt] = np.dot(R,OctaVoidSeed.T).T + displacements[disp_idx,:]
                cnt += 1
        self.voids = {}
        for key in OctaVoids.keys():
            VOID = OctaVoids[key]
            scaled_VOID, PBC_MAP = scale_void_get_PBCmap(VOID, cell_disp, Nxyz)
            self.voids[key] = {}
            self.voids[key]['OctaVoidPBC'] = np.copy(scaled_VOID)
            self.voids[key]['vertex_disp'] = np.copy(PBC_MAP)
            self.voids[key]['indeces'] = find_indeces(init_pos,scaled_VOID)
        self.number_voids = len(OctaVoids.keys())
        self.cell_position = np.copy(cell_disp)
        self._init_pos = np.copy(init_pos)
        
    def extract_void_coordinates(self, new_positions, new_lattice):
        voids = {}
        coords2unwrap = np.append(self._init_pos,new_positions,axis=0)
        unw = unwrap_PBC(coords2unwrap,self._init_pos.shape[0])
        new_coords_unw = unw[self._init_pos.shape[0]:,:]
        for key in self.voids.keys():
            direct_coord_void = self.voids[key]['OctaVoidPBC']
            disps = self.voids[key]['vertex_disp']
            indeces = self.voids[key]['indeces']
            void_direct_coords = new_coords_unw[indeces,:].reshape(direct_coord_void.shape[0],3) + disps 
            voids[key] = frac2cart(void_direct_coords,new_lattice)
        return voids
        
class OctaVoidType2():
    def __init__(self,cell_disp, Nxyz, init_pos):
        OctaVoidSeedType2 = np.array([[0,0,0],
                                      [0,1,0],
                                      [1,1,0],
                                      [1,0,0],
                                      [0.5,0.5,0.5],
                                      [0.5,0.5,-0.5]])
        generator = {}
        generator['XY'] = {}
        generator['XY']['rotation'] = [0,0,1]
        generator['XY']['angle'] = 0
        generator['XY']['shifts'] = np.array([[0,0,0],
                                            [0,0,1]])
        generator['YZ'] = {}
        generator['YZ']['rotation'] = [0,-1,0]
        generator['YZ']['angle'] = 90
        generator['YZ']['shifts'] = np.array([[0,0,0],
                                            [1,0,0]])
        generator['XZ'] = {}
        generator['XZ']['rotation'] = [1,0,0]
        generator['XZ']['angle'] = 90
        generator['XZ']['shifts'] = np.array([[0,0,0],
                                            [0,1,0]])
        cnt = 0
        OctaVoids = {}
        for plane in generator.keys():
            shifts = generator[plane]['shifts']
            rotation_dir = generator[plane]['rotation']
            rot_angle = generator[plane]['angle']
            for shift_idx in range(shifts.shape[0]):
                R = arbitrary_dir_rotation(rotation_dir,rot_angle)
                OctaVoids[cnt] = np.dot(R,OctaVoidSeedType2.T).T + shifts[shift_idx,:]
                cnt += 1
        self.voids = {}
        for key in OctaVoids.keys():
            VOID = OctaVoids[key]
            scaled_VOID, PBC_MAP = scale_void_get_PBCmap(VOID, cell_disp, Nxyz)
            self.voids[key] = {}
            self.voids[key]['OctaVoidPBC'] = np.copy(scaled_VOID)
            self.voids[key]['vertex_disp'] = np.copy(PBC_MAP)
            self.voids[key]['indeces'] = find_indeces(init_pos,scaled_VOID)
        self.number_voids = len(OctaVoids.keys())
        self.cell_position = np.copy(cell_disp)
        self._init_pos = np.copy(init_pos)
        
    def extract_void_coordinates(self, new_positions, new_lattice):
        voids = {}
        coords2unwrap = np.append(self._init_pos,new_positions,axis=0)
        unw = unwrap_PBC(coords2unwrap,self._init_pos.shape[0])
        new_coords_unw = unw[self._init_pos.shape[0]:,:]
        for key in self.voids.keys():
            direct_coord_void = self.voids[key]['OctaVoidPBC']
            disps = self.voids[key]['vertex_disp']
            indeces = self.voids[key]['indeces']
            void_direct_coords = new_coords_unw[indeces,:].reshape(direct_coord_void.shape[0],3) + disps 
            voids[key] = frac2cart(void_direct_coords,new_lattice)
        return voids 
############################################################################################################################
#
#               Plotting
#
############################################################################################################################ 
def plot_hist_pdf(DATA, kde_data, kde_args, init_vol = None, std_vol = None,nbins = 60, **kw):
    """
    A simple function to plot the histograms and the fitted data on top of it
    """
    # set general parameters
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    rc('axes', linewidth=2)
    mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
    tick_parameters_major = {
                'which': 'major',
                'direction': 'in',
                'width': 2,
                'length': 6,
                'labelsize': 14
            }
    tick_parameters_minor = {
        'which': 'minor',
        'direction': 'in',
        'width': 2,
        'length': 4,
        'labelsize': 14
    }
    labels_params = {
                'weight': 'bold',
                'size': 14
            }
    legend_params = {'loc': 'center', 'fontsize': 12}
    minor_ticks_num_x =kw.pop('minor_ticks_num_x',5) 
    minor_ticks_num_y = kw.pop('minor_ticks_num_y',5) 
    major_ticks_x = kw.pop('major_ticks_x',7) 
    major_ticks_y = kw.pop('major_ticks_y',5) 
    f_size = kw.pop('figure_size',(7,7))
    linewidth = kw.pop('linewidth',2)
    xlim = kw.pop('x_limits',None)
    
    fig, ax = plt.subplots(1,1, figsize = f_size)
    n, bins, patches = ax.hist(DATA,bins=nbins,density=False)
    vals_scaled = kde_data * max(n)
    ax.plot(kde_args,vals_scaled, lw=linewidth,c= 'r')
    if xlim != None:
        ax.set_xlim(xlim[0],xlim[1])
    ax.set_xlabel(r'\textbf{Volume (\AA$^{3}$)}', **labels_params)
    ax.set_ylabel(r'\textbf{Counts}',**labels_params)
    ax.grid()
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks_num_x))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks_num_y))
    #ax.xaxis.set_major_locator(MaxNLocator(major_ticks_x))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params('both', **tick_parameters_major)
    ax.tick_params('both', **tick_parameters_minor)
    ax.yaxis.set_major_locator(MaxNLocator(major_ticks_y))
    if init_vol != None:
        ax.axvline(init_vol, lw = linewidth, color = 'orange')
    ax.text(ax.get_xlim()[0]*1.1,ax.get_ylim()[1]*0.9, 'Mean %f, \nstd. %f' % (init_vol,std_vol), **labels_params)
    return fig
