import os
import numpy as np
from scipy import signal
import copy
import  multiprocessing
#
from VASP_common_utilites import *
from nearest_neighbours_A_B import *
from lib_progress_bar import *
from lib_plot2d_npt import * 
from lib_plot import  * 
import time
from lib_MP_bash import *
#
from matplotlib.backends.backend_pdf import PdfPages
from mayavi import mlab
from lib_plot_3d import * 
###################################################################
#  SIGNAL PROCESSING TOOLS
#
###################################################################
def runningMean(x, N):
    return signal.fftconvolve(x, np.ones((N,))/N,mode='same')
##############################################################################
#
#     Some operations on polymers' dictionary
#
###############################################################################
def NumChainsPerStep(C_C_bonds):
    NumChainsAllSteps = np.zeros((len(C_C_bonds.keys()),2))
    for j,key in enumerate(C_C_bonds.keys()):
        NumChains = 0
        for element in C_C_bonds[j]:
            if len(element)>2:
                NumChains +=1
        NumChainsAllSteps[j,0] = j 
        NumChainsAllSteps[j,1] = NumChains 
    return NumChainsAllSteps
#
def NumAtomsPerChain(C_C_bonds):
    NumAtomsAll=[]
    for j,key in enumerate(C_C_bonds.keys()):
        tmp = []
        for element in C_C_bonds[j]:
            if len(element)>2:
                tmp.append(len(element))
        NumAtomsAll.append(np.array(tmp))
    return NumAtomsAll
###################################################################
# PAIR CORRELATION FUNCTIONS
#
###################################################################
def correlate(dr_cart, CC_, the_pair,flag_abs = False):
    dr_0 = dr_cart[:,:,the_pair]
    corr__ = np.zeros((dr_0.shape[0],CC_.shape[0]), dtype=float)
    if flag_abs:
        for j in range(corr__.shape[1]):
            for i in range(corr__.shape[0]):
                corr__[i,j] = np.dot(dr_0[i,:],dr_cart[i,:,CC_[j]])
            corr__[:,j] = corr__[:,j]/np.max(corr__[:,j])
    else:
        for j in range(corr__.shape[1]):
            for i in range(corr__.shape[0]):
                corr__[i,j] = np.absolute(np.dot(dr_0[i,:],dr_cart[i,:,CC_[j]]))
            corr__[:,j] = corr__[:,j]/np.max(corr__[:,j])
    return corr__
###################################################################
#                       Sparsing a dataset
#
###################################################################
def sparse_data(dat, sparse_factor, spacing, third_daxis=False):
    """
    Function sparses the input data. If the data set
    is a 3D array, there is possibility to sparse 
    the third axis by sending third_axis=True
    spacing : number of atoms in each step
    """
    def sparse_data_3d(dat, sparse_factor, third_daxis):
        """
        accepts 3D array and sparses it
        """
        if not third_daxis:
            size = int(dat.shape[0] / sparse_factor)
            output = np.zeros((size, dat.shape[1], dat.shape[2]))
            for i in range(0, size):
                output[i, :, :] = dat[i * sparse_factor, :, :]
        if third_daxis:
            size = int(dat.shape[2] / sparse_factor)
            output = np.zeros((dat.shape[0], dat.shape[1], size))
            for i in range(0, size):
                output[:, :, i] = dat[:, :, i]
        return output

    def sparse_data_2d(dat, sparse_factor, spacing):
        """
        accepts 2D array and sparses it
        """
        size = int(dat.shape[0] / sparse_factor)
        n_steps = int(dat.shape[0] / spacing)
        output = np.zeros((size, dat.shape[1]))
        for step in range(0, n_steps):
            sep1_dat = step * spacing * sparse_factor
            sep2_dat = sep1_dat + spacing
            sep1_out = step * spacing
            sep2_out = sep1_out + spacing
            try:
                output[sep1_out:sep2_out, :] = np.copy(
                    dat[sep1_dat:sep2_dat, :])
            except:
                pass
        return output
    if sparse_factor == 1:
        return dat
    else:
        if len(dat.shape) == 3:
            output = sparse_data_3d(dat, sparse_factor, third_daxis)
        elif len(dat.shape) == 2:
            output = sparse_data_2d(dat, sparse_factor, spacing)
    return output

###################################################################
#                       CLASS DECLARATION
#
###################################################################
class ATOMS:
    def __init__(self, system_name, sc_atom, atom_labels, time_step, positions_fname, outcar_name,stress_fname, forces_name=None):
        print('Initialization')
        self.atom_labels = copy.deepcopy(atom_labels)
        self.system_name = system_name
        cwd = os.getcwd()
        dst = cwd+'/prepared_data/' + system_name+'/'
        try:
            os.mkdir(cwd+'/prepared_data/')
        except:
            pass
        try:
            os.mkdir(cwd+'/prepared_data/' + system_name)
        except:
            pass
        if '.dat' in positions_fname:
            positions_fname_core = positions_fname.split('/')[-1].replace('.dat','')
        else:
            positions_fname = positions_fname.split('/')[-1]+'.dat'
        # check if lattice parameters data is existing
        if os.path.isfile(dst+'__'+positions_fname_core+'_lattice.npy'):
            print('Loading lattice data')
            self.lattice = np.load(dst+'__'+positions_fname_core+'_lattice.npy')
            self.lattice_all_data = parse_outcar_lattice(outcar_name)
            print('Done')
            flag_read_lattice = False
        else:
            flag_read_lattice = True
        if flag_read_lattice:
            # check if OUTCAR exists
            if os.path.isfile(outcar_name):
                pass
            else:
                raise ValueError('OUTCAR is not found')
            print('No prepared NumPy format lattice data is found. Reading.')
            self.lattice_all_data = parse_outcar_lattice(outcar_name)
            self.lattice = mk_latt_3d_array (self.lattice_all_data)
            np.save(dst+'__'+positions_fname_core+'_lattice.npy',self.lattice)
        # check if the .npy of raw data exists
        fname2check_raw_cart = positions_fname.split('/')[-1].replace('.dat','') +'_raw_cart.npy'
        if os.path.isfile(dst +'__'+ fname2check_raw_cart):
            print('VASP positions (Cartesian) in NumPy format is found')
            self.raw_pos_cart = np.load(dst +'__'+ fname2check_raw_cart)
        else:
            print('No VASP positions (Cartesian) in NumPy format is found. Reading the .txt. It will take a while')
            self.raw_pos_cart = np.loadtxt(positions_fname)
            np.save(dst +'__'+ fname2check_raw_cart,self.raw_pos_cart)
            print('Done')
        # check if data in fractional coordinates exists:
        fname2check_raw_frac = positions_fname.split('/')[-1] +'_raw_frac.npy'
        if os.path.isfile(dst +'__'+ fname2check_raw_frac):
            print('VASP positions (fractional) in NumPy format is found')
            self.raw_pos_frac = np.load(dst +'__'+ fname2check_raw_frac)
        else:
            print('No VASP positions (fractional) in NumPy format is found')
            self.raw_pos_frac = cart2frac_2d_input(self.raw_pos_cart,self.lattice,sc_atom)
            print('Done. Saving.')
            np.save(dst +'__'+ fname2check_raw_frac,self.raw_pos_frac)
        #forces
        if forces_name == None:
            pass
        else:
            fname2check_forces = forces_name.split('/')[-1].replace('.dat','') +'_raw_cart.npy'
            if os.path.isfile(dst +'__'+ fname2check_forces):
                print('Forces in .npy are found')
                self.forces = np.load(dst +'__'+ fname2check_forces)
            else:
                print('No NumPy format forces are found. Reading the .dat')
                self.forces = np.loadtxt(forces_name)
                np.save(dst +'__'+ fname2check_forces,self.raw_pos_cart)
                print('Done')
        self.SC_Atoms = sc_atom
        self.file_pos_name = positions_fname_core
        self.cwd = cwd
        self.dst = dst
        self.num_steps = self.raw_pos_frac.shape[0]/self.SC_Atoms.sum()
        self.time_step = time_step
        self.time_range = np.arange(1,self.num_steps+1,1,dtype=int)*self.time_step
        if os.path.isfile(stress_fname):
            print('Reading stresses')
            self.stresses = np.loadtxt(stress_fname)
        else:
            print('No stresses were supplied!')
        print('Reading stresses')
        
        print('Initialization is finished')     

    def unwrap_PBC(self, flag_also_cart = False):
        #check if the unwrapped data exist
        if flag_also_cart:
            if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy'):
                self.cart_unwrapped = np.load(self.dst +'__'+ self.file_pos_name+'_unwrapped_cart.npy')                    
        if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_unwrapped.npy'):
            self.frac_unwrapped = np.load(self.dst +'__'+ self.file_pos_name+'_unwrapped.npy')
        else:
            print('No NumPy format data found. Performing unwrapping the periodic boundary conditions (PBC)')
            coords = self.raw_pos_frac
            N_atoms = self.SC_Atoms.sum()
            dim = coords.shape
            n_steps = dim[0] / N_atoms
            for step in range(0, n_steps - 1):
                progress(step+1, dim[0] / N_atoms)
                sep1_this = step * N_atoms
                sep2_this = sep1_this + N_atoms
                this_step = coords[sep1_this:sep2_this, :]
                sep1_next = sep2_this
                sep2_next = sep1_next + N_atoms
                next_step = coords[sep1_next:sep2_next, :]
                check_cond = next_step - this_step
                for atom in range(0, N_atoms):
                    for k in range(0, 3):
                        if check_cond[atom, k] > 0.5:
                            indexes = range((step + 1) * N_atoms +
                                            atom, dim[0], N_atoms)
                            coords[indexes, k] -= 1
                        elif check_cond[atom, k] < -0.5:
                            indexes = range((step + 1) * N_atoms +
                                            atom, dim[0], N_atoms)
                            coords[indexes, k] += 1
            print('\n')
            print('Done')
            self.frac_unwrapped = coords
            np.save(self.dst +'__'+ self.file_pos_name + '_unwrapped.npy',self.frac_unwrapped)
            if flag_also_cart:
                self.cart_unwrapped = frac2cart_2d_input(self.frac_unwrapped,self.lattice,self.SC_Atoms)
                np.save(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy',self.cart_unwrapped)
            
    def extract_atom_coordinates_2d(self,atoms_index,**kwargs):
        if len(kwargs) == 0:
            print('You have not specified what kind of coordinates to slice. Using default (fraction unwrapped)')
            if hasattr(self,'frac_unwrapped'):
                    pos = self.frac_unwrapped
            else:
                self.unwrap_PBC()
            pos = self.frac_unwrapped
        else:
            try:
                coord_kind = kwargs.pop('coord_type').lower()
            except:
                raise ValueError('You have not provided valid sort of coordinates. Valid are: "frac", "cart"')
            try:
                flag_unwrapped = kwargs.pop('unwrapped')
            except:
                print ('You have not specified wheather the coordinates are unwrapped. Assume ones with unwrapped PBC')
                flag_unwrapped = True
        if flag_unwrapped:
            if coord_kind == 'frac':
                if hasattr(self,'frac_unwrapped'):
                    pos = self.frac_unwrapped
                else:
                    self.unwrap_PBC()
                pos = self.frac_unwrapped
            if coord_kind == 'cart':
                if hasattr(self,'cart_unwrapped'):
                    pos = self.cart_unwrapped
                else:
                    if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy'):
                        self.cart_unwrapped = np.load(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy')
                    else:
                        print('Converting to fractional')
                        if hasattr(self, 'frac_unwrapped'):
                            self.cart_unwrapped = frac2cart_2d_input(self.frac_unwrapped,self.lattice,self.SC_Atoms)
                        else:
                            self.unwrap_PBC()
                            self.cart_unwrapped = frac2cart_2d_input(self.frac_unwrapped,self.lattice,self.SC_Atoms)
                        np.save(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy',self.cart_unwrapped)
                    pos = self.cart_unwrapped
        else:
            if coord_kind == 'frac':
                pos = self.raw_pos_frac
            if coord_kind == 'cart':
                pos = self.raw_pos_cart
        sc_at = self.SC_Atoms
        steps = pos.shape[0] / sc_at.sum()
        n_atoms_of_interest = pos.shape[0] / sc_at.sum() * sc_at[atoms_index]
        r = np.zeros((n_atoms_of_interest, 3))
        N_atoms = sc_at.sum()
        N_k_1 = 0
        if atoms_index > 0:
            for i in range(0, atoms_index):
                N_k_1 = N_k_1 + sc_at[i]
        else:
            N_k_1 = 0
        for i in range(1, steps + 1):
            index1 = (i - 1) * sc_at[atoms_index]
            index2 = i * sc_at[atoms_index]
            index1_pos = (i - 1) * sc_at.sum() + N_k_1
            index2_pos = (i - 1) * sc_at.sum() + N_k_1 + sc_at[atoms_index]
            r[int(index1):int(index2), 0:3] = pos[int(
                index1_pos):int(index2_pos), 0:3]
        return r
    
    def extract_atom_coordinates_3d(self,ind_atom,**kwargs):
        kw_copy = copy.deepcopy(kwargs)
        if len(kwargs) == 0:
            print('You have not specified what kind of coordinates to slice. Using default (fraction unwrapped)')
            pos = self.frac_unwrapped
        else:
            try:
                coord_kind = kwargs.pop('coord_type').lower()
            except:
                raise ValueError('You have not provided valid sort of coordinates. Valid are: "frac", "cart"')
            try:
                flag_unwrapped = kwargs.pop('unwrapped')
            except:
                print ('You have not specified wheather the coordinates are unwrapped. Assume ones with unwrapped PBC')
                flag_unwrapped = True
        if hasattr(self,'frac_unwrapped'):
            pass
        else:
            self.unwrap_PBC()
        if flag_unwrapped:
            if coord_kind == 'frac':
                if hasattr(self,'frac_unwrapped'):
                    pos = self.frac_unwrapped
                else:
                    self.unwrap_PBC()
                pos = self.frac_unwrapped
            if coord_kind == 'cart':
                if hasattr(self,'cart_unwrapped'):
                    pos = self.cart_unwrapped
                else:
                    if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy'):
                        self.cart_unwrapped = np.load(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy')
                    else:
                        print('Converting to fractional')
                        self.cart_unwrapped = frac2cart_2d_input(self.frac_unwrapped,self.lattice,self.SC_Atoms)
                    pos = self.cart_unwrapped
        else:
            if coord_kind == 'frac':
                pos = self.raw_pos_frac
            if coord_kind == 'cart':
                pos = self.raw_pos_cart
        dim = pos.shape
        sc_at = self.SC_Atoms
        n_steps = dim[0] / sc_at.sum()
        Ret = np.zeros((n_steps, 3, int(sc_at[ind_atom])))
        r1 = self.extract_atom_coordinates_2d(ind_atom,**kw_copy)       
        print('\n')
        for i in range(0, n_steps):
            progress(i, n_steps)
            for j in range(0, Ret.shape[2]):
                Ret[i, :, j] = r1[i * sc_at[ind_atom] + j]
        print('\n')
        return Ret
    
    def average_positions(self, skip=0):
        output = np.zeros((self.SC_Atoms.sum(),3))
        for step in range(skip,self.num_steps):
            output += extrac_all_coords_at_step(self.frac_unwrapped,self.SC_Atoms,step)
        output = output / (self.num_steps-skip)
        txt2poscar =  mk_text_for_POSCAR(self.system_name,self.atom_labels,1,self.SC_Atoms)
        fname='POSCAR_aver_skipped_' +str(skip) + '_'  + self.system_name
        write_POSCAR(fname,output,np.average(self.lattice[:,:,skip:], axis=2),txt2poscar)

    def get_step(self,step, kind = 'unwrapped'):
        if kind == 'unwrapped':
            if hasattr(self,'frac_unwrapped'):
                pos = self.frac_unwrapped
            else:
                self.unwrap_PBC()
                pos = self.frac_unwrapped
        if kind == 'PBC':
            pos = raw_pos_frac
        if kind != 'unwrapped' and kind != 'PBC':
            print('Unknown kind of coordinates, return raw VASP\'s output')
            pos = raw_pos_frac
        return extrac_all_coords_at_step(pos, self.SC_Atoms, step)
    
    def MD_snapshots(self,steps2extract=None):
        if steps2extract.all()==None:
            steps2extract = np.arange(0,self.num_steps,self.num_steps/10)
        cwd = os.getcwd()
        try:
            os.mkdir(cwd+ '/MD_snapshots')
        except:
            pass
        try:
                os.mkdir(cwd+ '/MD_snapshots/' + self.system_name)
        except:
            pass
        dst = cwd+ '/MD_snapshots/' + self.system_name
        for i in range(steps2extract.shape[0]):
            coords_step = extrac_all_coords_at_step(self.raw_pos_frac,self.SC_Atoms,steps2extract[i])
            txt2POSCAR = mk_text_for_POSCAR(self.system_name + str(steps2extract[i]),self.atom_labels,1,self.SC_Atoms)
            fname= dst + '/POSCAR_'+self.system_name+'_step_%d' %   steps2extract[i]
            write_POSCAR(fname,coords_step,self.lattice[:,:,steps2extract[i]],txt2POSCAR)
        #  
        lattice_list = []
        latt_vect_names = ['a','b','c']
        angles = ['alpha','beta','gamma']
        ratios = ['b/a','c/a']
        symbols = ['[',']',',']
        #
        cwd = os.getcwd()
        try:
            os.mkdir(cwd+ '/MD_snapshots')
        except:
            pass
        try:
                os.mkdir(cwd+ '/MD_snapshots/' + self.system_name)
        except:
            pass
        dst = cwd+ '/MD_snapshots/' + self.system_name
        #
        lattice_list.append('\t\t\t\tSystem: %s' % self.system_name)
        for i in range(steps2extract.shape[0]):
            lat_vect = self.lattice_all_data['step' + str(steps2extract[i]+1)]['lattice']
            lattice_list.append('Lattice parameters for step: %d' % steps2extract[i])
            lattice_list.append('\tVolume: %f' % self.lattice_all_data['step' + str(steps2extract[i]+1)]['volume'])
            lattice_list.append('\tLattice vector lengths:')
            lat_len = self.lattice_all_data['step' + str(steps2extract[i]+1)]['lat_vetc_len']
            for j in range(3):
                lattice_list.append('\t\t:' + latt_vect_names[j] + ': '+ '%f' % lat_len[j])
            lattice_list.append('\tAngles:')
            lat_angles =  self.lattice_all_data['step' + str(steps2extract[i]+1)]['angles']
            for j in range(3):
                lattice_list.append('\t\t:' + angles[j] + ': '+ '%f' % lat_angles[j])
            a = lat_vect[0,:]
            b = lat_vect[1,:]
            c = lat_vect[2,:]
            lattice_list.append('\tRatios:')
            for kk in range(2):
                ratio = lat_len[kk+1]/lat_len[0]
                lattice_list.append('\t\t:' + ratios[kk] + ': '+ '%f'% (ratio))
            lattice_list.append('\tLattice vectors:')
            for j in range(3):
                tmp = ''
                for k in range(3):
                    tmp += '\t'+"%.12f" %  lat_vect[j,k]
                for symbol in symbols:
                    tmp = tmp.replace(symbol,'')
                lattice_list.append('\t'+tmp)
            lattice_list.append('----------------------------------------------------------------------------')
        fname = dst + '/lattice_parameters.dat'
        #fname = 'lattice_params.dat'
        with open(fname,'w') as f:
            for line in lattice_list:
                f.write(line + '\n')
    
    def extract_forces_2d(self,atoms_index):
        pos = self.forces
        sc_at = self.SC_Atoms
        steps = pos.shape[0] / sc_at.sum()
        n_atoms_of_interest = pos.shape[0] / sc_at.sum() * sc_at[atoms_index]
        r = np.zeros((n_atoms_of_interest, 3))
        N_atoms = sc_at.sum()
        N_k_1 = 0
        if atoms_index > 0:
            for i in range(0, atoms_index):
                N_k_1 = N_k_1 + sc_at[i]
        else:
            N_k_1 = 0
        for i in range(1, steps + 1):
            index1 = (i - 1) * sc_at[atoms_index]
            index2 = i * sc_at[atoms_index]
            index1_pos = (i - 1) * sc_at.sum() + N_k_1
            index2_pos = (i - 1) * sc_at.sum() + N_k_1 + sc_at[atoms_index]
            r[int(index1):int(index2), 0:3] = pos[int(
                index1_pos):int(index2_pos), 0:3]
        return r
    
    def verlet_velocity_step(self,step,**kwargs):
        """
        calculates forces with Verlet algorithm
        Accepts positions of all atoms as output in OUTCAR. 
        """
        if len(kwargs) == 0:
            print('You have not specified what kind of coordinates to slice. Using default (fraction unwrapped)')
            pos = self.frac_unwrapped
        else:
            try:
                coord_kind = kwargs.pop('coord_type').lower()
            except:
                raise ValueError('You have not provided valid sort of coordinates. Valid are: "frac", "cart"')
            try:
                flag_unwrapped = kwargs.pop('unwrapped')
            except:
                print ('You have not specified wheather the coordinates are unwrapped. Assume ones with unwrapped PBC')
                flag_unwrapped = True
        if hasattr(self,'frac_unwrapped'):
            pass
        else:
            self.unwrap_PBC()
        if flag_unwrapped:
            if coord_kind == 'frac':
                if hasattr(self,'frac_unwrapped'):
                    pos = self.cart_unwrapped
                else:
                    self.unwrap_PBC()
                pos = self.frac_unwrapped
            if coord_kind == 'cart':
                if hasattr(self,'cart_unwrapped'):
                    pos = self.cart_unwrapped
                else:
                    if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy'):
                        self.cart_unwrapped = np.load(self.dst +'__'+ self.file_pos_name + '_unwrapped_cart.npy')
                    else:
                        print('Converting to fractional')
                        self.cart_unwrapped = frac2cart_2d_input(self.frac_unwrapped,self.lattice,self.SC_Atoms)
                    pos = self.cart_unwrapped
        else:
            if coord_kind == 'frac':
                pos = self.raw_pos_frac
            if coord_kind == 'cart':
                pos = self.raw_pos_cart
        N_atoms = self.SC_Atoms.sum()
        N_step = self.num_steps
        # indentify "begin" and "end" indexes to perform the array slicing
        # check if the target step is the 0th one: special considerations
        if step == 0:
            index_begin = 0
            index_end = N_atoms
            pos_t = pos[index_begin:index_end, :]
            pos_t_1 = pos[(index_begin + N_atoms):(index_end + N_atoms)]
            v = (pos_t - pos_t_1) / self.time_step
        elif step == N_step - 1:
            index_begin = (step - 1) * N_atoms
            index_end = step * N_atoms
            pos_t = pos[index_begin:index_end, :]
            pos_t_1 = pos[(index_begin - N_atoms):(index_end - N_atoms)]
            v = (pos_t - pos_t_1) / self.time_step
        elif step != N_step:
            t_m_1_begin = (step - 1) * N_atoms
            t_m_1_end = t_m_1_begin + N_atoms
            t_p_1_begin = step * N_atoms + N_atoms
            t_p_1_end = t_p_1_begin + N_atoms
            pos_t_minus_1 = pos[t_m_1_begin:t_m_1_end]
            pos_t_plus_1 = pos[t_p_1_begin:t_p_1_end]
            v = (pos_t_plus_1 - pos_t_minus_1) / (2 * self.time_step)
        return v

    def verlet_velocity_all_steps(self,**kwargs):        
        output = np.zeros(self.raw_pos_frac.shape)
        N_atoms = self.SC_Atoms.sum()
        N_steps = self.num_steps
        for st in range(1, N_steps + 1):
            output[(st - 1) * N_atoms:(st) * N_atoms,
                   :] = self.verlet_velocity_step(st-1,**kwargs)
            progress(st, N_steps, ' Calculating velocities')
        print('\n')
        self.velocities = output
        return output

    def mean_square_disp(self,**kwargs):
        kw_extract={
                'coord_type':'frac',
                'unwrapped':True,
            }
        def msd(at_coords, sc_at, ind_at_,**kwargs):
            """
            Mean squared displacement calculated for all atoms of the specified sort. 
            The result is expressed in units per atom
            (C) S. Filippov, June 8, 2017
            """
            flag_from_origin = kwargs.pop('flag_from_origin', False)
            print(flag_from_origin)
            if not flag_from_origin:
                dr = at_coords
                flag_from_origin = False
            else:                
                r0 = at_coords[0:sc_at[ind_at_], :]
                r0_v = np.tile(r0, (at_coords.shape[0] / (sc_at[ind_at_]), 1))
                dr = at_coords - r0_v
            steps = at_coords.shape[0] / (sc_at[ind_at_])
            msd1 = np.zeros(steps)
            for timestep in range(0, steps):
                sep1 = timestep * sc_at[ind_at_]
                sep2 = sep1 + sc_at[ind_at_]
                a = dr[sep1:sep2, :]
                tmp = np.linalg.norm(a,axis=1)**2
                msd1[timestep] = tmp.sum()
            if flag_from_origin:
                output = msd1 / sc_at[ind_at_]
            else:
                output = (msd1 - np.min(msd1)) / sc_at[ind_at_]
            return output
        msd_dict = {}
        for atom in range(len(self.atom_labels)):
            #get the positions
            atoms_positions = self.extract_atom_coordinates_2d(atom,**kw_extract)
            msd_dict[self.atom_labels[atom]] = msd(atoms_positions,self.SC_Atoms,atom,**kwargs)
        self.MSD = msd_dict
        return msd_dict
        
    def mk_data_for_hopping(self,hopp_sparse=10):
        kw_hopping={
                'coord_type':'cart',
                'unwrapped':False,}
        hopping_positions = {}
        init_positions = {}
        for items in range(0, len(self.SC_Atoms)):
            # extract the data for hopping
            tmp = self.extract_atom_coordinates_3d(items,**kw_hopping) 
            hopping_positions[str(items)] = sparse_data(tmp, hopp_sparse, self.SC_Atoms[items])
            # extrtact initial atomic positions
            sep1 = items * self.SC_Atoms[items]
            sep2 = sep1 + self.SC_Atoms[items]
            init_positions[str(items)] = np.copy(self.raw_pos_cart[sep1:sep2, :])
        self.hopping_positions = hopping_positions
        self.hopping_init_positions = init_positions
    
    def convert_pairs2spher(self,**kwargs):
        """
        converts the Cartesian coordinates into spherical with the polar axis 
        slecified in the ** kwargs ('pairs_aligned_along')
        """
        def vect_angle_abs(x, y):
            dot_prod = np.absolute(np.dot(x, y))
            norm_x = np.linalg.norm(x,axis = 1) #np.sqrt(np.dot(x, x.T))
            norm_y = np.linalg.norm(y,axis = 0) #np.sqrt(np.dot(y, y.T))
            angle = np.arccos(dot_prod / (norm_x * norm_y))
            return np.degrees(angle)       
        def cumulative_azimuth(y):
            """
            Unwraps the PBC for the azimuth angle and outputs increment vs the starting point            
            """
            out = np.zeros(y.shape[0])
            counter = 0
            for i in range(1,y.shape[0]):
                beta = y[i]
                alpha = y[i-1]
                if (np.absolute(alpha)/90 and np.absolute(beta)/90)>1:
                    if np.sign(beta)!=np.sign(alpha):
                        diff1 = beta - alpha
                        if beta<0:
                            diff2 = alpha - (360 + beta)
                            d_ang = min(np.absolute(diff1),np.absolute(diff2))
                        elif alpha<0:
                            diff2 = beta - (360 + alpha)
                            d_ang = - min(np.absolute(diff1),np.absolute(diff2))
                        counter += d_ang
                        out[i] =counter
                    else:
                        counter += beta - alpha
                        out[i] = counter
                else:
                    if i==1:
                        d_ang = 0
                    else:
                        d_ang = beta - alpha
                    counter += d_ang
                    out[i] = counter
            return out
        if hasattr(self,'dr_cart'):
            dr_rot = self.dr_cart
        else:
            raise ValueError('You must perform search for the pairs!')
        axis_aligned = kwargs.pop('pairs_aligned_along',np.array([0,0,1]))
        # perform conversion
        n_steps = self.num_steps
        dr_sph = np.ones(dr_rot.shape)
        steps_range = range(0, n_steps)
        N_pairs = range(0, dr_rot.shape[2])
        for pair in N_pairs:
            dr0 = - dr_rot[:,:,pair] / 2
            dr0_norm = np.linalg.norm(dr0, axis=1)
            dr_sph[:,0,pair] = np.linalg.norm(self.dr_cart[:,:,pair],axis=1,keepdims=True)[:,0] #dr
            #theta:
            dr_sph[:,1,pair] = vect_angle_abs(dr0,axis_aligned)
            # phi:
            dr_sph[:,2,pair] = cumulative_azimuth(np.degrees(np.arctan2(-dr0[:,1], -dr0[:,0]))) #phi
            progress(pair+1,dr_rot.shape[2],' Converting')
        self.dr_sph = dr_sph
        return dr_sph
        
    def get_polymers(self, atom_ind, nchunks=8, npools=8):
                #
        kw_PBC={
        'coord_type':'frac',
        'unwrapped':True,
        }
        stime = time.time()
        if hasattr(self,'_pos4chains'):
            coordinates3D = self._pos4chains
        else:
            print('Extracting the data into the convinient format')
            coordinates3D = self.extract_atom_coordinates_3d(atom_ind,**kw_PBC)
            self._pos4chains = np.copy(coordinates3D)
        # get first the coordination radii
        print('Building set of coordination radii')
        lattice  = self.lattice
        trans_vect = mk_translation_vector()
        coordination_orders = np.array([1,2])
        dbl_transpose = [False,True]
        AGRS = ((coordinates3D,lattice),(coordination_orders,trans_vect))
        JB = parallelize_framework(function=get_coordination_radii, _ARGS_=AGRS, 
                                   double_transpose_fix=dbl_transpose,nchunks=8, npools=8)
        print('Collecting the data')
        for (idx, element) in enumerate(JB):
            tmp = JB[idx]
            if idx == 0:
                coordinations = np.copy(tmp)
            else:
                coordinations = np.append(coordinations,tmp,axis=0)
        del JB
        # identify the polymeric/dimeric chains
        print('Searching for the bonds')
        dbl_transpose = [False,True, False]
        AGRS = ((coordinates3D,lattice,coordinations),(trans_vect,))
        JB = parallelize_framework(function=identify_polymers, _ARGS_=AGRS, 
                                   double_transpose_fix=dbl_transpose,nchunks=8, npools=8)
        print('Collecting the data')
        chains = collect_mp_output(JB)   
        self.polymers = copy.deepcopy(chains)
        del JB, chains
        print('Done in %f' % (time.time()-stime))
        return self.polymers

    def identify_pairs(self, dimer_ind_at = 1, ncpus = 8, canonic_pairs_list= None):
        """
        The call syntaxis is left for legacy purposes.
        This function is yet not trained to work in case of mixture of both dimmers and polymeric chains
        """
        # helper functions
        def remove_elements_with_wrong_length(lst, length_to_keep):
            """
            Removes all entries in a list of lists whose length
            differs from specified
            """
            _lst_ = copy.deepcopy(lst)
            remove_elements = []
            for (idx,element) in enumerate(_lst_):
                if len(element)!=length_to_keep:
                    remove_elements.append(element)
            for element in remove_elements:
                _lst_.remove(element)
            return _lst_,remove_elements
        # end of helper functions
        if hasattr(self,'polymers'):
            C_C = self.polymers
        else:
            C_C = self.get_polymers(atom_ind = dimer_ind_at ,nchunks = ncpus, npools = ncpus)
        # get the fractional uPBC coordinates
        C_coords_frac = self._pos4chains
        # set the C0 and C1 arrays
        self.C0 = np.zeros((len(C_C.keys()),3,len(C_C[0])))
        self.C1 = np.zeros(self.C0.shape)
        # the original list of the pairs (if not supplied)
        if canonic_pairs_list == None:
            canonic_pairs_list = C_C[0]
        # do the magic
        for step in range(0,len(C_C.keys())):
            # list of the atoms forming pairs, single atoms, chains 
            # at this step:
            new_pairs_list = C_C[step]
            # check if some of the pairs are broken into 
            # single atoms or combined into polymeric chains
            if len(new_pairs_list) != len(C_C[0]):
                # get the refined list of remaining pairs 
                # and the atoms no longer in the pairs
                new_pairs_list, rem_elements = remove_elements_with_wrong_length(new_pairs_list,2)
                # in this case we threat the the broken dimers
                # by keeping the C0 and C1 positions the
                # same as they were not broken
                for lst_element in rem_elements:
                    # since each lst_element might be a list itself, 
                    # it is wise to iterate over all possible elements
                    for element in lst_element:
                        for (canonic_idx,canonic_element) in enumerate(canonic_pairs_list):
                            try:
                                # by my convention, in the list of pairs [a,b]
                                # a corresponds to C0 (0-th element)
                                # b corresponds to C1 (1-st element)
                                if canonic_element.index(element) == 0:
                                    self.C0[step,:,canonic_idx] = self.C0[step-1,:,canonic_idx]
                                if canonic_element.index(element) == 1:
                                    self.C1[step,:,canonic_idx] = self.C1[step-1,:,canonic_idx]
                            except:
                                pass
            # work on the remaining pairs
            # or all the pairs, depending on situation above
            try:
                idx_list = []
                for pair in new_pairs_list:
                    idx_list.append(canonic_pairs_list.index(pair))
            except:
                continue
            for pair_idx in idx_list:
                C0 = C_coords_frac[step,:,canonic_pairs_list[pair_idx][0]]
                C1 = C_coords_frac[step,:,canonic_pairs_list[pair_idx][1]]
                # check that the C0 and C1 are not on the 
                # opposite sides of the simulation cell
                for k in range(3):
                    if (C1[k]-C0[k])> 0.5:
                        C1[k] -= 1
                    if (C1[k]-C0[k])< -0.5:
                        C1[k] += 1
                self.C0[step,:,pair_idx] = frac2cart(C0,self.lattice[:,:,step])
                self.C1[step,:,pair_idx] = frac2cart(C1,self.lattice[:,:,step])
            progress(step+1,len(C_C.keys()), 'Checking PBC')
        print('')
        self.dr_cart = self.C1 - self.C0
        return self.dr_cart  
        
    def correlate_all_pairs(self, min_bond, max_bond, **kw):
        """
        Correlate nearest neighbour pairs using dot product as correlation function.
        The method also filters (using Sav-Gol filter) the raw data
        """
        C0 = np.copy(self.C0)
        C1 = np.copy(self.C1)
        centers = (C0+C1)/2
        dr = self.dr_cart
        lattice = self.lattice
        dn_window = kw.pop('denoise_window',centers.shape[0]/100)
        polynom_order = kw.pop('polynomial_order',3)
        flag_absolut= kw.pop('absolute',False)
        if hasattr(self,'pair_correlation'):
            flag_only_filtered = True
        else:
            flag_only_filtered = False
        if not flag_only_filtered:
            start_i = kw.pop('start_step',0) # which step is consiederd as the first in the simulations
            centers_0step = np.round(cart2frac(centers[start_i,:,:].T, lattice[:,:,start_i]), decimals=6)
            self.CC_neighbours = find_neighbours_p(np.round(cart2frac(centers[start_i,:,:].T, lattice[:,:,start_i]), decimals=6),lattice[:,:,start_i],min_bond, max_bond)
            self.pair_correlation = {}
            self.pair_correlation['step0'] = centers_0step
            for i in range(len(self.CC_neighbours)):
                self.pair_correlation[i] = {}
                # convert the C-C neighbours to array
                _CC_ = np.array(list(self.CC_neighbours[i]))
                self.pair_correlation[i]['neighbours'] = np.copy(_CC_)
                self.pair_correlation[i]['original_positions'] = centers_0step[np.array(list(self.CC_neighbours[i])),:]
                self.pair_correlation[i]['correlation'] = {}
                self.pair_correlation[i]['correlation']['raw'] = correlate(dr,_CC_,i,flag_abs = flag_absolut)
                progress(i+1,len(self.CC_neighbours),'Correlating the pairs with unfiltered coordinates')
            dr_smoothed = savgol_filter(dr, window_length=(dn_window/2)*2+1,polyorder=polynom_order,axis=0)
            print('\n')
            print('Filtering the dr vectors with Sav-Gol filter with %d window of %d order' % ((dn_window/2)*2+1, polynom_order))
            for i in range(len(self.CC_neighbours)):
                _CC_ = np.array(list(self.CC_neighbours[i]))
                self.pair_correlation[i]['correlation']['filtered'] = correlate(dr_smoothed,_CC_,i,flag_abs = flag_absolut)
                progress(i+1,len(self.CC_neighbours),'Correlating the pairs with filtered coordinates')
        else:
            print('Filtering the dr vectors with Sav-Gol filter with %d window of %d order' % ((dn_window/2)*2+1, polynom_order))
            dr_smoothed = savgol_filter(dr, window_length=(dn_window/2)*2+1,polyorder=polynom_order,axis=0)
            for i in range(len(self.CC_neighbours)):
                _CC_ = np.array(list(self.CC_neighbours[i]))
                self.pair_correlation[i]['correlation']['filtered'] = correlate(dr_smoothed,_CC_,i,flag_abs = flag_absolut)
                progress(i+1,len(self.CC_neighbours),'Correlating the pairs with filtered coordinates')
        return self.pair_correlation

    def correlate_pairs(self,**kwargs):
        """
        Identifies the correlation between 2 selected pairs. They are specified in **kwargs
        with keys 'pair1' and 'pair2'
        
        """
        def correlate_opt(x,y,tstep):
            if x.shape != y.shape:
                raise ValueError('Dimensions of X and Y must be the same')
            corr = np.zeros(x.shape)
            N = x.shape[0]
            for j in range(0,N):
                corr[j]=np.sum(x[0:N-j-1]*y[j:N-1],axis=0)/((N-j)*tstep)
                progress(j,N,' computing correlation function')
            return corr    
        if len(kwargs)==0:
            raise ValueError('You have not supplied any information for correlation')
        else:
            try:
                pair1 = kwargs.pop('pair1')
            except:
                raise ValueError('You have not specified any pair which has to be correlated!')
            try:
                pair2 = kwargs.pop('pair2')
            except:
                print('You have no specified a pair to correlated with. Conseder auto-correlation')
                pair2 = pair1
        if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_corr_'+str(pair1)+'_'+str(pair2) +'.npy'):
            print('Found pre-calculated data. Reading.')
            correlation = np.load(self.dst +'__'+ self.file_pos_name + '_corr_'+str(pair1)+'_'+str(pair2) +'.npy')
        else:
            if hasattr(self,'dr_cart'):
                print('No precalculated data found. Computing.')
                correlation = correlate_opt(self.dr_cart[:,:,pair1],self.dr_cart[:,:,pair2],self.time_step)
                np.save(self.dst +'__'+ self.file_pos_name + '_corr_'+str(pair1)+'_'+str(pair2),correlation)
            else:
                raise ValueError('You have to identify the bonds before you compute the correlation!')
        return correlation
    
    
    def show_angles(self, skip_steps=None):
        """
        Prints avereaged angels between the lattice vectors for set of steps to be skipped
        """
        angles = break_dict_by_key_2d(self.lattice_all_data,'angles')
        angles_lbls = ['alpha','beta','gamma']
        out_lst = []
        if skip_steps==None:
            skip_steps = np.arange(0,self.num_steps+1, 1000,dtype=int)
        print('Averaged agnles for the %s system' % self.system_name)
        print('======================================================')
        out_lst.append('======================================================')
        out_lst.append('Averaged agnles for the %s system' % self.system_name)
        for j in range(0,skip_steps.shape[0]):
            print('Skipping %d steps:' % skip_steps[j])
            out_lst.append('Skipping %d steps:' % skip_steps[j])
            for i in range(3):
                line = '\t'+angles_lbls[i] + ': %d'  % np.average(angles[skip_steps[j]:,i])
                print(line)
                out_lst.append(line)
            print('-------------------------------------------------------')
            out_lst.append('-------------------------------------------------------')
        dst = self.cwd + '/plots/' + self.system_name + '/angles_log.dat'
        with open(dst,'w') as f:
            for line in out_lst:
                f.write(line + '\n')

    
    def plot_polymer_hist(self,**kwargs):
        def P_Chains_2(C_C_bonds):
            """
            Return a NumPy array with indeces of atoms forming th polymeric chains  at each time step
            """
            dist = np.array([])
            for j,key in enumerate(C_C_bonds.keys()):
                for i,element in enumerate(C_C_bonds[j]):
                    if len(element)>2:
                        dist = np.append(dist, np.array(element))                        
            return dist        
        """
        Plots the histogram
        
        """
        if hasattr(self, 'polymers'):
            C_C_chains = self.polymers
        else:
            raise ValueError('You should identify the polymeric chains first!')
        dist = P_Chains_2(C_C_chains)+1
        x_label_increment = kwargs.pop('x_label_increment',2)
        minor_ticks_num_x = x_label_increment
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        rc('axes', linewidth=1.75)
        mpl.rcParams['text.latex.preamble'] = [r'\boldmath']
        tick_parameters_major = {
            'which': 'major',
            'direction': 'in',
            'width': 1,
            'length': 5.5,
            'labelsize': 14
        }
        tick_parameters_minor = {
            'which': 'minor',
            'direction': 'in',
            'width': 1,
            'length': 3,
            'labelsize': 14
        }
        labels_params = {
            'weight': 'bold',
            'size': 18
        }
        legend_params = {'loc': 'best', 'fontsize': 12}
        annotate_params = {'size': 18, 'weight': 'bold'}
        fig, axs = plt.subplots(1, 1, figsize = (10,10))
        l = axs.hist(dist,normed=None,bins=self.SC_Atoms[1])
        axs.grid()
        #
        axs.set_xlim(left = 1, right = self.SC_Atoms[1])
        axs.set_xticks(np.arange(1,self.SC_Atoms[1]+1,x_label_increment))
        #
        axs.minorticks_on()
        axs.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks_num_x))
        axs.yaxis.set_ticks_position('both')
        axs.xaxis.set_ticks_position('both')
        axs.tick_params('both', **tick_parameters_major)
        axs.tick_params('both', **tick_parameters_minor)
        axs.set_xlabel('Atom index',**labels_params)
        axs.set_ylabel('Number of timesteps in a chain',**labels_params)
        #
        title = self.system_name
        axs.set_title(title, **annotate_params)
        return fig,axs
    
    
    def plot_number_of_chains_atoms_per_chain(self):
        if hasattr(self, 'polymers'):
            C_C_chains = self.polymers
        else:
            raise ValueError('You should identify the polymeric chains first!')
        fig, ax = plt.subplots(2,1, sharex = True, figsize = (10,10), gridspec_kw = {'wspace':0, 'hspace':0})
        NumAtPerChain = NumAtomsPerChain(C_C_chains)
        NumChainsAllSteps = NumChainsPerStep(C_C_chains)
        #
        tick_parameters_major = {
            'which': 'major',
            'direction': 'in',
            'width': 1,
            'length': 5.5,
            'labelsize': 14
        }
        tick_parameters_minor = {
            'which': 'minor',
            'direction': 'in',
            'width': 1,
            'length': 3,
            'labelsize': 14
        }
        labels_params = {
            'weight': 'bold',
            'size': 16
        }
        legend_params = {'loc': 'best', 'fontsize': 12}
        annotate_params = {'size': 18, 'weight': 'bold'}
        #
        for i in range(0,len(NumAtPerChain)):
            x = np.ones(NumAtPerChain[i].shape[0],dtype=int)*i
            if x.shape[0]!=0:
                ax[0].scatter(x,NumAtPerChain[i], c='r', s=0.5)
            else:
                ax[0].scatter(i,0,c='r',s=0.5)
        ax[0].grid()
        #
        ax[1].plot(NumChainsAllSteps[:,0],NumChainsAllSteps[:,1])
        ax[1].grid()
        #
        ax[0].set_ylabel('Number of atoms per chain',**labels_params)
        ax[0].set_title('System: %s' % self.system_name)
        ax[1].set_ylabel('Number of polymeric chains',**labels_params)
        ax[1].set_xlabel('Number of time steps',**labels_params)
        #
        for i in range(2):
            ax[i].tick_params('both', **tick_parameters_major)
            ax[i].tick_params('both', **tick_parameters_minor)
        return fig,ax
    
    def get_time_polymerized(self):
        """
        Identifies how much of the time steps were 
        in the polymerized state
        """
        if not hasattr(self,'polymers'):
            raise ValueError('You should identify the polymeric chains frist!')
        else:
            ChainsPerStep = NumChainsPerStep(self.polymers)
            condition = (ChainsPerStep[:,1] == 0.0)
            NoPolymers = ChainsPerStep[condition,1].shape[0]
            return (ChainsPerStep.shape[0] - NoPolymers) / float(ChainsPerStep.shape[0])

    def get_initialization_time(self):
        """
        Identifies when the first polymerization event happens
        """
        if not hasattr(self,'polymers'):
            raise ValueError('You should identify the polymeric chains frist!')
        else:
            ChainsPerStep = NumChainsPerStep(self.polymers)
            condition = (ChainsPerStep[:,1] == 0.0)
            return list(condition).index(False)
    
    def get_average_polymerized_atoms(self):
        """
        Returns average number of atoms involved in the polymerization
        averaged over all simulation time
        """
        if not hasattr(self,'polymers'):
            raise ValueError('You should identify the polymeric chains frist!')
        else:
            NumChainsStep = NumAtomsPerChain(self.polymers)
            polymer_step = 0
            average_num_atoms = 0
            for (idx,item) in enumerate(NumChainsStep):
                if len(NumChainsStep[idx]) != 0:
                    average_num_atoms += NumChainsStep[idx].sum()
                    polymer_step += 1
            return average_num_atoms / float(polymer_step)
            
    def get_polymerized_atoms(self):
        """
        Returns time evolution of total number of atoms
        polymerized at each time step
        """
        if not hasattr(self,'polymers'):
            raise ValueError('You should identify the polymeric chains frist!')
        else:
            NumChainsStep = NumAtomsPerChain(self.polymers)
            output = np.zeros((len(NumChainsStep),2))
            output[:,0] = np.arange(1, len(NumChainsStep)+1)
            for (idx,item) in enumerate(NumChainsStep):
                if len(NumChainsStep[idx]) == 0:
                    output[idx,1] = 0
                else:
                    output[idx,1] = NumChainsStep[idx].sum()
            return output
        
    def write_all_pairs(self,fname,**kwargs):
        print_figures_multi_pdf(self.time_range,self.dr_sph,fname,self.cwd,self.system_name,**kwargs)
        print('Done')
        
    def plot_selected_pairs(self,plot_index,**kwargs):
        fig,ax = plot_pairs(self.time_range,self.dr_sph,plot_index,**kwargs)
        return fig, ax
        
    def write_all_atoms_hopping(self,hopping_name_lst,atom_labels):
        if hasattr(self,'hopping_init_positions'):
            pass
        else:
            self.mk_data_for_hopping
        init_pos_dict = self.hopping_init_positions
        plt_dat_for_hopping_dict = self.hopping_positions
        lattice_init = self.lattice[:, :, 0]
        # plot first type of atoms
        hopping_name = hopping_name_lst[0]
        flag_show_Li = True
        num_figs_per_side = 2
        plt.ioff()
        print('Plotting')
        print_figures_multi_hopping_pdf(init_pos_dict, plt_dat_for_hopping_dict, lattice_init, self.SC_Atoms,
                                        flag_show_Li, self.cwd, hopping_name, num_figs_per_side, atom_labels, self.system_name)
        # plot second type of atoms
        plt.ioff()
        hopping_name = hopping_name_lst[1]
        flag_show_Li = False
        print_figures_multi_hopping_pdf(init_pos_dict, plt_dat_for_hopping_dict, lattice_init, self.SC_Atoms,
                                        flag_show_Li, self.cwd, hopping_name, num_figs_per_side, atom_labels, self.system_name)
        print('Done')
        
    def mk_hopping3d_data(self):
        kw_hopping={
            'coord_type':'cart',
            'unwrapped':False,}
        # init positions:
        if hasattr(self,'C0'):
            C0_i = self.C0[0,:,:].T
            C1_i = self.C1[0,:,:].T
        else:
            raise ValueError('You must perfrom the bond search first')
        Li_i = self.extract_atom_coordinates_2d(0,**kw_hopping)[0:self.SC_Atoms[0],:]
        BBox = mk_cell_box_npt(self.lattice)
        if hasattr(self,'Li_pos_3d'):
            Li_pos_arr = self.Li_pos_3d
        else:
            Li_pos_arr = self.extract_atom_coordinates_3d(0,**kw_hopping)
            self.Li_pos_3d = np.copy( Li_pos_arr)
        return C0_i,C1_i,Li_i,BBox, Li_pos_arr

    def plot_diffusion_3d(self,Li_num_atoms,CC_num_pairs,**kwargs):
        C0_i,C1_i,Li_i,BBox, Li_pos_arr = self.mk_hopping3d_data()
        start = kwargs.pop('Li_steps_start',0)
        end = kwargs.pop('Li_steps_stop',self.num_steps)
        Li_pos_arr_submit = Li_pos_arr[start:end,:,:]
        fig = plot_hopping_3d(C0_i,C1_i,self.C0,self.C1,Li_i,BBox, Li_pos_arr_submit,self.lattice,Li_num_atoms,CC_num_pairs,**kwargs)
        return fig        
        
    def average_pairs_(self):
        output = np.zeros((self.dr_sph.shape [2],3))
        n_steps = self.dr_sph.shape[0]
        for pair in range(self.dr_sph.shape[2]):
            output[pair,0] = self.dr_sph[:,0,pair].sum() / n_steps #dr
            output[pair,1] = self.dr_sph[:,1,pair].sum() / n_steps #theta
            output[pair,2] = self.dr_sph[:,2,pair].sum() / n_steps #phi
        self.averaged = output
        return output
    
    def plot_averaged(self,skip_steps = 0):
        if skip_steps == 0:
            aver_dat = np.average(self.dr_sph,axis=0).T
            std = np.std(self.dr_sph,axis=0).T
        else:
            aver_dat = np.average(self.dr_sph[skip_steps:-1,:,:],axis=0).T
            std = np.std(self.dr_sph[skip_steps:-1,:,:],axis=0).T
        fig,ax = plot_averaged(aver_dat,self.system_name)  
        self.averaged = np.copy(aver_dat)
        self.std = np.copy(std)
        return fig, ax
        
    def plot_msd(self):
        if hasattr(self,'MSD'):
            fig,ax = plot_msd(self.MSD, self.time_range, self.system_name)
        else:
            MSD = self.mean_square_disp()
            fig,ax = plot_msd(MSD, self.time_range, self.system_name)
        return fig,ax
         
    def plot_stresses(self,stress_fname, **kwargs):
        stresses = np.loadtxt(stress_fname)
        time = self.time_range
        fig, ax = plot_stresses(self.time_range,stresses,self.system_name)
        return fig,ax
    
    def plot_lattice_evolution(self):
        fig,ax = plot_lattice(self.lattice_all_data, self.time_step, self.system_name)
        return fig,ax
    
    def plot_3d_with_pairs(self,time_step,**kwargs):
        """
        this 3D plotting method assumes presence of 2 different kinds of atoms
        with atoms of the second type forming pairs. 
        """
        kw_extract_Li_cart_uPBC={
            'coord_type':'cart',
            'unwrapped':True,
        }
        kw_verlet_velocity = {
            'coord_type':'cart',
            'unwrapped':True,
        }
        fig_size = kwargs.pop('fig_size',(1024,1024))
        fig = mlab.figure(1, bgcolor=(0.105, 0.105, 0.105), size=fig_size)
        # prepare the data
        if hasattr(self,'sc_box'):
            pass
        else:
            self.sc_box = mk_cell_box_npt(self.lattice)
        if hasattr(self,'C0'):
            pass
        else:
            raise ValueError('You have to perform search of the pairs!')
        if hasattr(self,'C1'):
            pass
        else:
            raise ValueError('You have to perform search of the pairs!')        
        if hasattr(self,'Li_for_3d_plot'):
            Li_atoms_2d_cart = self.Li_for_3d_plot
        else:
            if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_Li_atoms_2d_cart.npy'):
                Li_atoms_2d_cart = np.load(self.dst +'__'+ self.file_pos_name + '_Li_atoms_2d_cart.npy')
            else:
                Li_atoms_2d_cart = self.extract_atom_coordinates_2d(0,**kw_extract_Li_cart_uPBC)
                np.save(self.dst +'__'+ self.file_pos_name + '_Li_atoms_2d_cart',Li_atoms_2d_cart)
                self.Li_for_3d_plot = np.copy(Li_atoms_2d_cart)
        if hasattr(self,'v_Li_for_3dplot'):
            v_Li = self.v_Li_for_3dplot
        if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_v_Li.npy'):
            v_Li = np.load(self.dst +'__'+ self.file_pos_name + '_v_Li.npy')
        else:
            v_Li = self.verlet_velocity_all_steps(**kw_verlet_velocity)
            np.save(self.dst +'__'+ self.file_pos_name + '_v_Li',v_Li)
            self.v_Li_for_3dplot = np.copy(v_Li)
        #plot
        fig, plt_dict = render_scene(fig, self.C0, self.C1, Li_atoms_2d_cart, v_Li, time_step,
                                     self.SC_Atoms,self.sc_box, self.lattice, **kwargs)
        #mlab.close()
        return
    
    def mk_batch_render(self,numtasks,Fname_core,sparse_factor=1):
        flag_show_bonds = True
        flag_show_flow = True
        lib_list = ['lib_progress_bar.py', 'lib_plot_3d.py', 'render.py']  # list of libraries to copy
        Var_Names = ['C0', 'C1', 'Li_atoms_pos', 'v_Li',
                     'sc_box', 'SC_Atoms', 'lattice_data']
        #
        kw_extract_Li_cart_uPBC={
            'coord_type':'cart',
            'unwrapped':True,
        }
        kw_verlet_velocity = {
            'coord_type':'cart',
            'unwrapped':True,
        }
        if hasattr(self,'sc_box'):
            pass
        else:
            self.sc_box = mk_cell_box_npt(self.lattice)
        if hasattr(self,'Li_for_3d_plot'):
            Li_atoms_2d_cart = self.Li_for_3d_plot
        else:
            if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_Li_atoms_2d_cart.npy'):
                Li_atoms_2d_cart = np.load(self.dst +'__'+ self.file_pos_name + '_Li_atoms_2d_cart.npy')
            else:
                Li_atoms_2d_cart = self.extract_atom_coordinates_2d(0,**kw_extract_Li_cart_uPBC)
                np.save(self.dst +'__'+ self.file_pos_name + '_Li_atoms_2d_cart',Li_atoms_2d_cart)
                self.Li_for_3d_plot = np.copy(Li_atoms_2d_cart)
        if hasattr(self,'v_Li_for_3dplot'):
            v_Li = self.v_Li_for_3dplot
        if os.path.isfile(self.dst +'__'+ self.file_pos_name + '_v_Li.npy'):
            v_Li = np.load(self.dst +'__'+ self.file_pos_name + '_v_Li.npy')
        else:
            v_Li = self.verlet_velocity_all_steps(**kw_verlet_velocity)
            np.save(self.dst +'__'+ self.file_pos_name + '_v_Li',v_Li)
            self.v_Li_for_3dplot = np.copy(v_Li) 
        # sparse the data
        sc_box=sparse_data(self.sc_box,sparse_factor,self.sc_box.shape[0],True)
        C0 = sparse_data(self.C0, sparse_factor, self.SC_Atoms[1], False)    
        C1 = sparse_data(self.C1, sparse_factor, self.SC_Atoms[1], False)
        Li_atoms_batch = sparse_data(Li_atoms_2d_cart, sparse_factor, self.SC_Atoms[0], False)
        v_Li_batch = sparse_data(v_Li,sparse_factor,self.SC_Atoms[0], False)
        lattice_batch = sparse_data(self.lattice,sparse_factor,self.lattice.shape[0], True)
        START=0 #makes no sense, but it is easier to keep
        STOP=10 #makes no sense, but it is easier to keep
        flag_show_bonds=True #makes no sense, but it is easier to keep
        flag_show_flow=True #makes no sense, but it is easier to keep
        # make arguments
        args_ = mk_arguments(C0, C1, Li_atoms_batch, v_Li_batch, sc_box, self.SC_Atoms, lattice_batch, START, STOP,
                             self.cwd, self.system_name, Fname_core, flag_show_bonds, flag_show_flow, numtasks)
        # make folders
        mk_arguments_folders(args_, Var_Names)
        copy_libs(args_, lib_list)
        # make launch scripts
        mk_launch_script(args_)
        mk_batch_launch_script(args_)
