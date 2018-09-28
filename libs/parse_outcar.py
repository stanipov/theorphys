import numpy as np
from scipy import linalg
import os
import re
import time

import cPickle as pickle

from VASP_common_utilites import *

###################################################################
#
#  SIMPLE PROGRESS BAR
#
###################################################################
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def parse_poscar(p_name):
    poscar_list = []
    with open(p_name, 'r') as f:
        for line in f:
            poscar_list.append(str(line).replace('\n', ''))
    try:
        NumAtoms = np.array(poscar_list[6].split(),dtype=int)
        name_atoms = poscar_list[5].split()
    except:
        NumAtoms = np.array(poscar_list[5].split(),dtype=int)
        name_atoms = []
    return NumAtoms, name_atoms
class OUTCAR:
    def __init__(self, o_name='OUTCAR', p_name = 'POSCAR', system_name = '', reparse = False):
        cwd = os.getcwd()
        self.NumAtoms, self.name_atoms = parse_poscar(p_name)   
        if system_name == '' and not self.name_atoms:
            system_name = time.asctime(time.localtime(time.time())).replace(' ','_').replace(':','-')
        dst = cwd + '/' + system_name
        rePosition = re.compile(r"\s*POSITION\s*TOTAL-FORCE.*")
        if not reparse:
            print('Read saved data')
            self.positions_raw_cart = np.load(dst + '/positions_raw_cart.npy')
            self.coordinates_direct = np.load(dst + '/coordinates_direct.npy')
            self.forces = np.load(dst + '/forces.npy')
            self.lattice = np.load(dst + '/lattice.npy')
            with open(dst + '/lattice_dict.pkl', 'rb') as handle:
                self.lattice_dict = pickle.load(handle)
            self.NumAtoms = np.load(dst + '/NumAtoms.npy',)
            with open(dst + '/name_atoms.pkl', 'rb') as handle:
                self.name_atoms = pickle.load(handle)
            print('Done')
        else:
            pos_force_lst = []
            print('Parsing OUTCAR')
            print('Parsing positions and forces...')
            counter = 1
            with open(o_name, 'r') as f:
                while 1:
                    line=f.readline()
                    if not line: break
                    if(rePosition.match(line)):
                        counter += 1
                        f.readline()
                        for i in range(0,self.NumAtoms.sum()):
                            try:
                                pos_force_lst.append(str(map(float,f.readline().strip().split())))
                            except:
                                print(counter,i)
            tmp_arr = list2array(pos_force_lst)
            self.positions_raw_cart = tmp_arr[:,0:3]
            self.forces = tmp_arr[:,3:]
            print('Parsing lattice...')
            self.lattice_dict = parse_outcar_lattice(o_name)
            self.lattice = mk_latt_3d_array(self.lattice_dict)
            self.coordinates_direct = cart2frac_2d_input(self.positions_raw_cart, self.lattice, self.NumAtoms)
            try:
                os.mkdir(dst)
            except:
                pass
            print('Saving')
            fname =  dst + '/' + '_self_outcar.pkl'
            np.save(dst + '/positions_raw_cart.npy',self.positions_raw_cart)
            np.save(dst + '/coordinates_direct.npy',self.coordinates_direct)
            np.save(dst + '/forces.npy',self.forces)
            np.save(dst + '/lattice.npy',self.lattice)
            with open(dst + '/lattice_dict.pkl', 'wb') as handle:
                pickle.dump(self.lattice_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(dst + '/NumAtoms.npy',self.NumAtoms)
            with open(dst + '/name_atoms.pkl', 'wb') as handle:
                pickle.dump(self.name_atoms, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Done')
            
    def unwrap_PBC(self):
        if hasattr(self, 'coordinates_direct_uPBC'):
            return self.coordinates_direct_uPBC
        else:
            self.coordinates_direct_uPBC = unwrap_PBC(self.coordinates_direct, self.NumAtoms)
            return self.coordinates_direct_uPBC
    
    def get_direct_uPBC_3d(self, atom_idx):
        if hasattr(self, 'coordinates_direct_uPBC'):
            return extract_atom_coordinates_3d(self.coordinates_direct_uPBC,self.NumAtoms,atom_idx)
        else:
            coords = self.unwrap_PBC()
            return extract_atom_coordinates_3d(self.coordinates_direct_uPBC,self.NumAtoms,atom_idx)
    
    def get_direct_PBC_3d(self, atom_idx):
        return extract_atom_coordinates_3d(self.coordinates_direct,self.NumAtoms,atom_idx)
    
    def get_direct_PBC_2d(self, atom_idx):
        return extract_atom_coordinates_2d(self.coordinates_direct,self.NumAtoms,atom_idx)
    
    def get_direct_uPBC_2d(self, atom_idx):
        if hasattr(self, 'coordinates_direct_uPBC'):
            return extract_atom_coordinates_3d(self.coordinates_direct_uPBC,self.NumAtoms,atom_idx)
        else:
            coords = self.unwrap_PBC()
            return extract_atom_coordinates_2d(self.coordinates_direct_uPBC,self.NumAtoms,atom_idx)
