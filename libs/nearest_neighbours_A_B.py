import numpy as np
from scipy import linalg
import sys
import time
#
import multiprocessing as mp
import  multiprocessing

#
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
##############################################################################
#
#     Finds all atomsB which are neighbors to atomsA
#
##############################################################################
def coordination_order_distance(norm,coordination_order):
    """
    Returns a radius of a i-th coordination sphere
    Inputs:
    --norm: 2D array of norms of (atomic_positions - coordinates of j-th atom)
    --coordination_order: number of the coordination sphere
    Output:
    -- sphere radius (float)
    """
    _norm_ = np.copy(norm)
    _bond_ = 0
    for i in range(1,coordination_order+1):
        _norm__i = _norm_-_bond_
        idxs = np.argwhere(_norm_ > 0.00)
        _bond_ += np.min(_norm_[idxs[:,0],idxs[:,1]]) 
    return _bond_

def mk_translation_vector():
    """
    Creates an array with rows as translation vectors.
    These vectors represent all neighbouring cell to account the PBC
    """
    sc_dim = np.array([2, 2, 2])
    output = np.zeros((27, 3))
    counter = 0
    for i in np.arange(-1, 2):
        for j in np.arange(-1, 2):
            for k in np.arange(-1, 2):
                output[counter, :] = np.array([i, j, k])
                counter += 1
    return output
#
def extend_cell_by_tr_vect(pos_step, tr_vect,latt):
    """
    Translates the supplied 2D array along all posible directions
    determined by the "tr_vect". Direct (fractional) lattice coordinates
    are assumed.
    Inputs:
    -- pos_step: a 2D array of direct coordinates (at a time step)
    -- tr_vect: the translation vector
    -- latt: lattice 2d (3x3) matrix
    Output:
    -- a 3D array of Cartesian coordinates [<atoms>,xyz,<translated by a row from tr_vect>]
    """
    output = np.zeros((pos_step.shape[0], pos_step.shape[1], tr_vect.shape[0]))
    for (i, vect) in enumerate(tr_vect):
        output[:, :, i] = np.dot((pos_step + vect),latt)
    return output

def find_all_neighbours(posA, posB,lattice,min_bond, max_bond):
    """
    A general algorithm to search bonds between atoms A and B (in this order!)
    posA(B) -- fractional (direct) coordinates of the atoms A(B)
    lattice -- NumPy array of the lattice vectors
    min(max)_bond -- minimal(maximal) distance between the atoms
    RETURNS:
    a dictionary bonds_dict with the strucutre:
    |--'indeces': all possible inxeing is here
    |   |--'absolut' -- indexes in the concatenated array of posA, posB (in this sequence)
    |   |--'only_atoms_A(B)' -- indeces of the atoms A only in the array posA(B)
    |--'distance'
        |--'only_atoms_A(B)' --- distances between atoms A(B)
    | 'dr' -- Cartesian displacemnts of all the neighbours relatively to an atom
    """
    coords = np.append(posA, posB,axis=0)
    translation_vect = mk_translation_vector()
    extended_cell = extend_cell_by_tr_vect(coords,translation_vect,lattice)
    # indexes of neighbours of given atom
    bonds_dict = {}
    for AtomI in range(0,posA.shape[0]):
        dr_extended_cell = extend_cell_by_tr_vect(coords-coords[AtomI,:],translation_vect,lattice)
        dr_norm = np.linalg.norm(dr_extended_cell, axis=1,keepdims=True)
        nn_arr = np.argwhere(np.logical_and(dr_norm>= min_bond, dr_norm <= max_bond))
        bonds_dict[AtomI] = {}
        bonds_dict[AtomI]['indeces'] = {}
        bonds_dict[AtomI]['indeces']['absolut'] = nn_arr[:,0]
        bonds_dict[AtomI]['indeces']['only_atoms_A'] = nn_arr[:,0][nn_arr[:,0]<=posA.shape[0]-1]
        bonds_dict[AtomI]['indeces']['only_atoms_B'] = nn_arr[:,0][nn_arr[:,0]>=posA.shape[0]-1]-posA.shape[0]
        bonds_dict[AtomI]['distances'] = {}
        flag_AA = nn_arr[:,0]<=posA.shape[0]-1
        flag_BB = nn_arr[:,0]>=posA.shape[0]-1
        bonds_dict[AtomI]['distances']['only_atoms_A'] = dr_norm[nn_arr[:,0][flag_AA],:,nn_arr[:,2][flag_AA]]
        bonds_dict[AtomI]['distances']['only_atoms_B'] = dr_norm[nn_arr[:,0][flag_BB],:,nn_arr[:,2][flag_BB]]
        bonds_dict[AtomI]['dr'] = np.copy(dr_extended_cell[nn_arr[:,0],:,nn_arr[:,2]])
    return bonds_dict

def find_neighbours(coords,_lattice_, translation_vect):
    """
    This function generates a graph with indexes of the nearest neighbors of each atom
    each vertex contains indeces of its neighbors.
    Inputs:
    -- coords: a set of coordinates in direct _lattice_, 2D
    -- _lattice_: a matrix with _lattice_ vectors in rows
    --translation_vect: a amtrix with rows of translation vectors to the nearest cells
    Output:
    -- a graph with indeces of all nearest neighbours of each atom of the input
    """
    #translation_vect = mk_translation_vector()
    graph = {}
    for AtomI in range(0,coords.shape[0]):
        dr_extended_cell = extend_cell_by_tr_vect(coords-coords[AtomI,:],translation_vect,_lattice_)
        dr_norm = np.linalg.norm(dr_extended_cell, axis=1)
        first_coordination_dist = coordination_order_distance(dr_norm,1)
        second_coordination_dist = coordination_order_distance(dr_norm,2)
        # account for possible numerical errors:
        distance_min = first_coordination_dist*0.99
        # consider only half-distance between first and second coordination
        # spheres, so each atom is not counted twice
        distance_max = second_coordination_dist-first_coordination_dist*0.5
        nn_ = np.argwhere(np.logical_and(dr_norm >= distance_min, dr_norm < distance_max))
        graph[AtomI] = {}
        graph[AtomI] = set(nn_[:,0])
    return graph

def get_coordination_radii(_coordinates_,_lattice_,_coordination_orders_,trans_vect):
    """
    Returns coodination radii and the norm of the displaced cell for each atom and time _step_
    """
    coordinations = np.zeros((_coordinates_.shape[0],_coordinates_.shape[2],_coordination_orders_.shape[0]))
    for _step_ in range(_coordinates_.shape[0]):
        for AtomI in range(_coordinates_.shape[2]):
            _dr_tmp = extend_cell_by_tr_vect((_coordinates_[_step_,:,:].T - _coordinates_[_step_,:,AtomI]),trans_vect,_lattice_[:,:,_step_])
            _dr_tmp_norm = np.linalg.norm(_dr_tmp,axis=1)
            for coord_idx in range(_coordination_orders_.shape[0]):
                coordinations[_step_,AtomI,coord_idx] = coordination_order_distance(_dr_tmp_norm,_coordination_orders_[coord_idx])
    return coordinations

def identify_polymers(_coordinates_,_lattice_,_coordinations_,trans_vect):
    """
    A smart way to idintify polymeric/dymerics chains based on usage on first and 
    second coordination spheres radii. In case if an abrupt bond break happens,
    a the minimal and maximal distances are corrested with respect of the average 
    coordination radii.
    """
    output = {}
    for step in range(_coordinates_.shape[0]):
        graph = {}
        coords = _coordinates_[step,:,:].T
        _lattice_step = _lattice_[:,:,step]
        for atom_idx in range(_coordinates_.shape[2]):
            dr_extended_cell = extend_cell_by_tr_vect(coords-coords[atom_idx,:],trans_vect,_lattice_step)
            dr_step_atom = np.linalg.norm(dr_extended_cell, axis=1)
            # identify minimum distance and maximum distances
            # first coordination sphere radii
            first__coordinations_ = _coordinations_[:,atom_idx,0]
            # second coordination sphere radii
            second__coordinations_ = _coordinations_[:,atom_idx,1]
            # average values
            first_coordination_aver = np.average(first__coordinations_)
            second_coordination_aver = np.average(second__coordinations_)
            # standard deviation
            first_coordination_std = np.std(first__coordinations_)
            second_coordination_std = np.std(first__coordinations_)
            # minumim distance
            min_distance  = first__coordinations_[step] - first_coordination_std
            if min_distance > first_coordination_aver:
                min_distance = first_coordination_aver - first_coordination_std
            # maximum distance:
            if second__coordinations_[step] >=second_coordination_aver:
                _dr_aver = second_coordination_aver - first_coordination_aver
                max_distance =  first_coordination_aver + _dr_aver/2
            else:
                max_distance = first__coordinations_[step] + ( second__coordinations_[step] - first__coordinations_[step])/2
            nn_ = np.argwhere(np.logical_and(dr_step_atom >= min_distance, dr_step_atom < max_distance))
            if nn_.shape[0] == 0:
                graph[atom_idx] =  set([atom_idx])
            else:
                graph[atom_idx] =  set(nn_[:,0])
        output[step] = connected_components(graph)
    return output

def identify_polymers_old(coords_3d,_lattice_,start_i=0):
    """
    Identifies polymeric chains of atoms of the same element.
    Inputs:
    -- coords_3d: a 3D array of coordinates in a form of [steps,:,:]
    -- _lattice_: a 3D array of lattice vectors in a form of [:,:,steps]
    -- start_i: optionall, one cane specify the first index in the output dictionary
    Output:
    -- a dictionary with indeces of nearest neighbours or polymeric chains
    """
    output_dict = {}
    translation_vect = mk_translation_vector()
    for i in range(0,coords_3d.shape[0]):
        gr = find_neighbours(coords_3d[i,:,:].T, _lattice_[:,:,i], translation_vect)
        output_dict[i+start_i] = connected_components(gr)
    return output_dict

def collect_mp_output(res_dict):
    """
    A simple function to merge a dictionary of dictionaries
    into one dictionary
    """
    output_dict = {}
    cnt = 0
    for (idx, key) in enumerate(res_dict.keys()):
        chunk_dict = res_dict[idx]
        for i in range(len(chunk_dict.keys())):
            output_dict[cnt] = chunk_dict[i] #copy.deepcopy(chunk_dict[i])
            cnt += 1
    return output_dict

def complexes_extract(pos_A,pos_B,lattice,ligands_per_center,min_AB,max_AB, silent=True):
    """
    Extracts dr=ligands_for_pos_A(AtomNum)-pos_A(AtomNum) for each MD time step.
    The function assumes that for each time step number of ligands for each central atom A
    is constant and all A atoms have the same number of ligands
    """
    num_steps = pos_A.shape[0]
    NH_complexes = np.zeros((num_steps, ligands_per_center, 3))
    AB_dr_dict = {}
    for i in range(pos_A.shape[2]):
        AB_dr_dict[i] = np.copy(NH_complexes)
    if silent:
        for step in range(num_steps):
            pos_A_snapshot = pos_A[step,:,:].T
            pos_B_snapshot = pos_B[step,:,:].T
            bonds_dict_AB = find_all_neighbours(pos_A_snapshot,pos_B_snapshot,lattice[:,:,step],min_AB,max_AB)
            for AtomNum in range(len(bonds_dict_AB.keys())):
                AB_dr_dict[AtomNum][step,:,:] = bonds_dict_AB[AtomNum]['dr']
    else:
        for step in range(num_steps):
            pos_A_snapshot = pos_A[step,:,:].T
            pos_B_snapshot = pos_B[step,:,:].T
            bonds_dict_AB = find_all_neighbours(pos_A_snapshot,pos_B_snapshot,lattice[:,:,step],min_AB,max_AB)
            for AtomNum in range(len(bonds_dict_AB.keys())):
                AB_dr_dict[AtomNum][step,:,:] = bonds_dict_AB[AtomNum]['dr']
            progress(step+1,num_steps)
    return AB_dr_dict

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
    
def parallelize_framework(function, _ARGS_, double_transpose_fix, nchunks = 4, npools = 4):
    """
    This is a simple framework to parallelize a function using Python's
    inbuilt class Multiprocessing.Poll.
    Inputs:
    -- function: name of the function to be parellelized (proved with 1 returning value)
    -- _ARGS_ : a tuple of arguments. 
        IMPORTANT! _ARGS_ has structure _ARGS_ = ((args0),(args1)).
        args0 -- a tuple of arguments of 3D arrays (like time series of coordinates)
        args1 -- a tuple of scalar arguments, like a bond length
        IMPORTANT! shapes of args0 must be the SAME! (naturally, in fact)
    -- double_transpose_fix: a list of False or True values for EACH argument in args0.
        IMPORTANT! By default, args0 are assumed to be of shape [time steps, :, :]. 
        However if some of the arguments in args0 of the "function" should have
        shape [:,:,timesteps], then it has to be brought to shape of time steps, :, :] and then back.
        This means the argument should be transposed befor sliced and transformed back again, thus
        doulbly transposed.
    -- nchunks: number of chunks for the parallelization
    -- npools : number of workers in the pool to be created
    OUTPUT:
    -- a dictionary with keys in order of the arguments chunks. You have to correctly assemble your data from it
    """
    # make pool of workers
    _pool_ = mp.Pool(processes=npools)
    # submit the jobs
    Nstart = 0
    Nstop = _ARGS_[0][0].shape[0] 
    step = (Nstop-Nstart) / nchunks
    jobs = {}
    start_time = time.time()
    print('Submitting %d chunks to %d workers' % (nchunks, npools))
    for index in range(nchunks):
        starti = Nstart+index*step
        # account for possibility that Nstart+(index+1)*step might be < Nstop
        # and then we miss some of the data points
        if index == nchunks-1:
            endi = Nstop
        else:
            endi = min(Nstart+(index+1)*step, Nstop)
        arg_list = []
        for (IDX,arg_) in enumerate(_ARGS_[0]):
            if double_transpose_fix[IDX]:
                tmp = arg_.T[starti:endi,:,:]
                arg_list.append(tmp.T)
            else:
                arg_list.append(arg_[starti:endi,:,:])
        for arg_ in _ARGS_[1]:
            arg_list.append(arg_)
        job = _pool_.apply_async(func=function,args=tuple(arg_list))
        jobs[index] = job
    job_result = {}
    # collect the results
    for idx in range(len(jobs)):
        job_result[idx] = jobs[idx].get()
    print('Time elapsed %f sec' % (time.time() - start_time))
    #terminate the pool to avoid memory leaks
    _pool_.terminate()
    return job_result
