import numpy as np
from scipy import linalg
import os
import re
import linecache
import sys
import time

import copy
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
###################################################################
#
#  Text to array conversion
#
###################################################################
def extrat_num_from_list(lst):
    """
    Returns a NumPy 1D array from a list with mixed
    numerical and text elements
    """
    ff = []
    for element in lst:
        try:
            ff.append(float(element))
        except:
            pass
    return np.array(ff)


def list2array(out_lst):
    """
    Converts a list, which has elements with mixed extries of numbers and digits
    into a 2D NumPy array
    """
    tmp = extrat_num_from_list(out_lst[0].replace('\n','').replace(',','').replace('[','').replace(']','').split())
    out_array = np.zeros((len(out_lst),tmp.shape[0]))
    for row in range(out_array.shape[0]):
        out_array[row,:] = extrat_num_from_list(out_lst[row].replace('\n','').replace(',','').replace('[','').replace(']','').split())
    return out_array


def parse_stresses(path, fname):
    fopen = path + '/' + fname
    out_lst = []
    with open (fopen, 'r') as f:
        for line in f:
            if 'Total+' in line:
                out_lst.append(line)
    return list2array(out_lst)
###################################################################
#
#  PARSE OUTCAR FOR LATICE
#
###################################################################
def parse_outcar_lattice(fname):
    step = 0
    tmp_ar = np.zeros((3, 3))
    angles = np.zeros((3, 1))
    vlen = np.zeros((3, 1))
    latt_dict = {}
    with open(fname) as f:
        for (i, line) in enumerate(f):
            if 'VOLUME and BASIS-vectors are now' in line:
                step += 1
                key = 'step' + str(step)
                latt_dict[key] = {}
                for (k, j) in enumerate(range(i + 6, i + 9)):
                    tmp_ar[k, :] = np.asarray(str(linecache.getline(fname, j)).replace(
                        '\n', '').split()[0:3], dtype=float)
                latt_dict[key]['lattice'] = np.copy(tmp_ar)
                for kk in range(0, 3):
                    vlen[kk] = np.sqrt(np.dot(tmp_ar[kk, :], tmp_ar.T[kk, :]))
                latt_dict[key]['lat_vetc_len'] = np.copy(vlen)
                latt_dict[key]['volume'] = float(
                    str(linecache.getline(fname, i + 4)).replace('\n', '').split()[4])
                angles[0] = vect_angle(
                    tmp_ar[2, :], tmp_ar[1, :])  # c*b, i.e. alpha
                angles[1] = vect_angle(
                    tmp_ar[2, :], tmp_ar[0, :])  # c*a, i.e. beta
                angles[2] = vect_angle(
                    tmp_ar[0, :], tmp_ar[1, :])  # a*b, i.e. gamma
                latt_dict[key]['angles'] = np.copy(angles)
    linecache.clearcache()
    return latt_dict

def mk_latt_3d_array(latt_dict):
    """
    Extraction of lattice parameters sotred into a dictionary
    into a 3D array with Z axis corresponding to to each step of MD
    """
    n_steps = len(latt_dict.keys())
    output = np.zeros((3, 3, n_steps))
    print('Exctrating lattice')
    for index in range(0, n_steps):
        key = 'step' + str(index + 1)
        output[:, :, index] = np.copy(latt_dict[key]['lattice'])
        progress(index+1, n_steps, '')
    print('\n')
    return output

def vect_angle(x, y):
    """
    Returns angle between 2 angles
    """
    dot_prod = np.dot(x, y)
    norm_x = np.sqrt(np.dot(x, x.T))
    norm_y = np.sqrt(np.dot(y, y.T))
    angle = np.arccos(dot_prod / (norm_x * norm_y))
    return np.degrees(angle)
###################################################################
#
#  COORDINATES MANIPULATIONS
#
###################################################################
def cart2frac_2d_input(pos_pbc, lattice, sc_atoms):
    """
    Conversion of cartesian coordinates as read from OUTCAR to fractional
    performed with concern of variable lattice.
    """
    output = np.zeros(pos_pbc.shape)
    sep1 = 0
    sep2 = 0
    n_atoms = sc_atoms.sum()
    n_steps = pos_pbc.shape[0] / n_atoms
    for index in range(0, n_steps):
        sep1 = int(index * n_atoms)
        sep2 = int(sep1 + n_atoms)
        output[sep1:sep2, :] = np.copy(
            cart2frac(pos_pbc[sep1:sep2, :], lattice[:, :, index]))
        progress(index+1, n_steps, '')
    print('\n')
    return output

def frac2cart_3d_input(pos, lattice):
    pos_cart = np.zeros(pos.shape)
    for i in range(pos.shape[0]):
        pos_cart[i,:,:] = frac2cart(pos[i,:,:].T,lattice[:,:,i]).T
    return pos_cart

def cart2fract_3d_input(pos, lattice):
    pos_cart = np.zeros(pos.shape)
    for i in range(pos.shape[0]):
        pos_cart[i,:,:] = cart2fract(pos[i,:,:].T,lattice[:,:,i]).T
    return pos_cart


def frac2cart_2d_input(pos_pbc, lattice, sc_atoms):
    """
    Conversion of fractional coordinates to cartesian ones
    performed with concern of variable lattice.
    """
    n_atoms = sc_atoms.sum()
    n_steps = pos_pbc.shape[0] / n_atoms
    output = np.zeros(pos_pbc.shape)
    sep1 = 0
    sep2 = 0
    for index in range(0, n_steps):
        sep1 = intfrac2cart_npt(index * n_atoms)
        sep2 = int(sep1 + n_atoms)
        output[sep1:sep2, :] = np.copy(
            frac2cart(pos_pbc[sep1:sep2, :], lattice[:, :, index]))
        progress(index+1, n_steps, '')
    print('\n')
    return output

def cart2frac(dataset, a_lat):
    """
    This function covrets the cartesian atomic coordinates into fractional ones.
    """
    a_inv = linalg.inv(a_lat)
    return np.dot(dataset,a_inv)

def frac2cart(dataset, a_lat):
    """
    This function covrets the fractions atomic coordinates into cartesian ones.
    Returns:
            a 2d NumPy array of Cartesian coordites
    """
    return np.dot(dataset,a_lat)

def unwrap_PBC(coords,num_atoms):
    """
    Unwrapping the PBC put on the MD data
    """
    N_atoms = num_atoms.sum()
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
    return coords

def extract_atom_coordinates_2d(pos, sc_at, atoms_index):
    """
    Extracts coordinates to a 2D array of a specific atomic spicies.
    Inputs:
    -- pos: a 2D array of positions for all time steps
    -- sc_at: 1D array with numbers of each atomic species
    -- atoms_index: index of atomic spicies (in the sc_at array) to extract
    Output:
    -- a 2D NumPy array
    """
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
    
def extract_atom_coordinates_3d(pos,sc_at,ind_atom):
    """
    Extracts coordinates to a 3D array of a specific atomic spicies.
    Inputs:
    -- pos: a 2D array of positions for all time steps
    -- sc_at: 1D array with numbers of each atomic species
    -- ind_atom: index of atomic spicies (in the sc_at array) to extract
    Output:
    -- a 3D NumPy array with [steps, 3, number_of_atoms]
    """
    dim = pos.shape
    n_steps = dim[0] / sc_at.sum()
    Ret = np.zeros((n_steps, 3, int(sc_at[ind_atom])))
    r1 = extract_atom_coordinates_2d(pos,sc_at,ind_atom)       
    print('\n')
    for i in range(0, n_steps):
        progress(i, n_steps)
        for j in range(0, Ret.shape[2]):
            Ret[i, :, j] = r1[i * sc_at[ind_atom] + j]
    print('\n')
    return Ret

def implement_PBC (data__):
    """
    Implement of the PBC on the supplied 2D array (elementwise!)
    """
    data = np.copy(data__)
    for i in range(data.shape[0]):
        for j in range(3):
            if data[i,j]>=1:
                data[i,j]-=1
            if data[i,j]<0:
                data[i,j]+=1
    return data
    
def segregate_by_planes(positions, plane_idx,tolerance=1e-02):
    """
    Returns a list with indeces of atoms
    lying in the same planes (XY, ZY,XZ) 
    defined by plane_idx:
    -- XY : 0
    -- YZ : 1
    -- XZ : 2
    The input data is 2D!
    """
    if np.ndim(positions) > 2:
	raise ValueError('The input coordinates must be 2D array!')
    else:
	idx_dict = {}
	for i in range(positions.shape[0]):
	    condition = np.isclose(positions[:,plane_idx] -positions[i,plane_idx],0,atol=tolerance)
	    idx_dict[i] = set(np.argwhere(condition==True).flatten()) # build a connected graph
	return connected_components(idx_dict)
    
def round_positions(_pos_,tolerance=1e-2):
    """
    Forces the atoms to be in the nearest XY,YZ, XZ plane
    """
    output = np.zeros(_pos_.shape, dtype=float)
    for plane_idx in range(3):
	idx_list = segregate_by_planes(_pos_,plane_idx, tolerance=tolerance)
	for j in range(len(idx_list)):
	    idx_ = idx_list[j]
	    plane_idx_coords = _pos_[idx_][:,plane_idx]
	    common_coord = np.round(min(plane_idx_coords*(1/tolerance*10)))/(1/tolerance*10)
	    for i in range(len(idx_)):
	        output[idx_[i],plane_idx] = common_coord
    return output
###################################################################
#
#  WRITING DOWN POSCAR
#
###################################################################
def mk_text_for_POSCAR(sys_name, at_lst_names, sc_factor, SC_atoms):
    text_to_POSCAR = {}
    text_to_POSCAR['system_name'] = sys_name
    text_to_POSCAR['atoms_names_list'] = at_lst_names
    text_to_POSCAR['scaling_factor'] = str(sc_factor)
    text_to_POSCAR['number_atoms'] = SC_atoms
    return text_to_POSCAR

def write_POSCAR(fname, pos_step, lat_data, text_to_POSCAR,flag_direct = True):
    with open(fname, 'w') as f:
        f.write(text_to_POSCAR['system_name'] + '\n')
        f.write(str(text_to_POSCAR['scaling_factor']) + '\n')
        # append lattice
        np.savetxt(f, lat_data, fmt='%22.18f')
        # append atoms names and their numbers
        tmp = str(text_to_POSCAR['atoms_names_list'])
        symbols_to_replace = ['[', ']', ',', '\'']
        for i in range(0, len(symbols_to_replace)):
            tmp = tmp.replace(symbols_to_replace[i], '')
        f.write(tmp + '\n')
        symbols_to_replace = ['[', ']']
        tmp = str(text_to_POSCAR['number_atoms'])
        for i in range(0, len(symbols_to_replace)):
            tmp = tmp.replace(symbols_to_replace[i], '')
        f.write(tmp + '\n')
        # append positions
        if flag_direct:
            f.write('Direct' + '\n')
        else:
            f.write('Cartesian' + '\n')
        np.savetxt(f, pos_step, fmt='%22.18f')
    return
    
###################################################################
#
#  PARSE POSCAR FOR ATOMS LABELS AND NUMBERS
#
###################################################################    
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
##############################################################################
#
#     Find connected components in the graph
#
##############################################################################
def connected_components(graph):
    """
    This finds connected components in undirected graph, which must be provided in symmetric form 
    (both edge x to y and y to x always given)
    The graph is specified as dict:
    graph[vertex] is set(all edges of vertex)
    
    """
    neighbors = graph
    seen = set()
    def component(node):
        queue = set([node])
        result = list()
        while queue:
            node = queue.pop()
            seen.add(node)
            #for x in neighbors[node]:
            #    if x not in seen:
            #        queue.append(x)
            queue |= set(neighbors[node]) - seen
            result.append(node)
        return result
    all_result = list()
    for node in neighbors:
        if node not in seen:
            all_result.append(component(node))
    return all_result
