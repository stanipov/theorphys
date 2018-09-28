import numpy as np
from scipy import linalg
import sys
#
import multiprocessing as mp
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
def extend_cell_by_tr_vect_dev(pos_step, tr_vect,latt):
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
    extended_cell = extend_cell_by_tr_vect_dev(coords,translation_vect,lattice)
    # indexes of neighbours of given atom
    bonds_dict = {}
    for AtomI in range(0,posA.shape[0]):
        dr_extended_cell = extend_cell_by_tr_vect_dev(coords-coords[AtomI,:],translation_vect,lattice)
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
    _pool_ = mp.Pool(processes=npools)
    # submit the jobs
    Nstart = 0
    Nstop = _ARGS_[0][0].shape[0] 
    step = (Nstop-Nstart) / nchunks
    jobs = {}
    for index in range(nchunks):
        starti = Nstart+index*step
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
    return job_result
