import numpy as np
import os
import copy


def exctrac_UC_coordinates_from_SC(array__, Nxyz):
    """
    Extracts the UCs from a SC. The SC is assumed to have the 
    atoms on their ideal positions, i.e. it is a typically the starting
    structure. 
    Inputs:
    -- array__: input array of positions of the ideal cell in direct coordinates
    -- Nxyz: dimensions of the SC in UCs (e.g. 2x2x2)
    Outputs:
    -- uc_from_sc_dict: a dictionary with indeces of atoms in the input array which form a UC
    -- dUC: a dictionary with position of the UC in the SC(e.g. 1,1,0 means it is
        2nd along X, second along Y, and the first along Z UC)
    """
    cnt = 0
    uc_from_sc_dict = {}
    array = np.copy(array__)
    # conditions
    condition_less_x = np.linspace(1.0/Nxyz[0],1.0, num = Nxyz[0] , dtype=float)
    condition_more_x = np.linspace(1.0/Nxyz[0],1.0, num = Nxyz[0] , dtype=float) - 1.0 / Nxyz[0]
    condition_less_y = np.linspace(1.0/Nxyz[1],1.0, num = Nxyz[1] , dtype=float)
    condition_more_y = np.linspace(1.0/Nxyz[1],1.0, num = Nxyz[1] , dtype=float) - 1.0 / Nxyz[1]
    condition_less_z = np.linspace(1.0/Nxyz[2],1.0, num = Nxyz[2] , dtype=float)
    condition_more_z = np.linspace(1.0/Nxyz[2],1.0, num = Nxyz[2] , dtype=float) - 1.0 / Nxyz[2]
    # coordinates of the UC:
    d_x = condition_less_x * Nxyz[0] - 1
    d_y = condition_less_x * Nxyz[1] - 1
    d_z = condition_less_x * Nxyz[2] - 1
    # coordinates of the UC, output:
    dUC = {}
    for i in range(Nxyz[0]):
        for j in range(Nxyz[1]):
            for k in range(Nxyz[2]):
                idx_less_x = np.argwhere(array[:,0] < condition_less_x[i])
                idx_more_x = np.argwhere(array[:,0] >= condition_more_x[i])
                idx_less_y = np.argwhere(array[:,1] < condition_less_y[j])
                idx_more_y = np.argwhere(array[:,1] >= condition_more_y[j])            
                idx_less_z = np.argwhere(array[:,2] < condition_less_z[k])
                idx_more_z = np.argwhere(array[:,2] >= condition_more_z[k])
                idx_x = np.intersect1d(idx_less_x,idx_more_x)
                idx_y = np.intersect1d(idx_less_y,idx_more_y)
                idx_z = np.intersect1d(idx_less_z,idx_more_z)
                idx_xy = np.intersect1d(idx_x,idx_y)
                idx_xyz = np.intersect1d(idx_xy,idx_z)
                uc_from_sc_dict[cnt] = np.copy(idx_xyz)
                dUC[cnt] =  np.array([d_x[i] / Nxyz[0], d_y[j] / Nxyz[1],d_z[k] / Nxyz[2]])
                cnt +=1
    return uc_from_sc_dict, dUC

def extract_H_at_centers(dict_,arr_idxs,arr_centers):
    """
    A helper to get coordinates of the ligands 
    around their equilibrium positions.
    Inputs:
    -- dict_: a dictionarry with coordinates (Cartesian!) of the
        ligands with respect to thier centers for each time step
    -- arr_idxs: indeces of these centers
    -- arr_centers: Cartesian coordinates of all the censters
    """
    for i in range(arr_idxs.shape[0]):
        idx = arr_idxs[i]
        dH_coords = dict_[idx]['dr']
        if i == 0 :
            H_N_cart = arr_centers[i,:] + dH_coords
        else:
            H_N_cart = np.append(H_N_cart,arr_centers[i,:] + dH_coords, axis=0)
    return H_N_cart

def impose_selective_dyn_on_part(p_name,coord_line_idx,immobile_num, new_fname):
    """
    A trivial procedure to make POSCAR for selective dynamics. It assumes that
    first "immobile_num" of atoms should be totally immobile ('F F F' in POSCAR),
    while the rest can be moved by VASP. 
    coord_line_idx: indes of a line where the first atomic coordinate is
    """
    p_lst = []
    with open(p_name,'r') as f:
        for line in f:
            p_lst.append(line.replace('\n',''))
    new_p_lst = copy.deepcopy(p_lst[0:coord_line_idx-1])
    new_p_lst.append('Selective dynamics')
    new_p_lst.append(p_lst[coord_line_idx-1])
    for idx, line in enumerate(p_lst[coord_line_idx:]):
        if idx < immobile_num:
            new_p_lst.append( p_lst[coord_line_idx+idx] + ' F F F')
        else:
            new_p_lst.append( p_lst[coord_line_idx+idx] + ' T T T')
    with open(new_fname,'w') as f:
        for line in new_p_lst:
            f.write(line + '\n')
    return
