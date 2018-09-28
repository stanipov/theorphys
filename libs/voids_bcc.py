"""
A simple library with classes representing various voids in a BCC 
(body centered cubic) structure.
"""
import numpy as np
import os
import copy
#
from VASP_common_utilites import *
#
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

class TetraVoid():      
    def __init__(self,cell_disp, Nxyz, lattice, init_pos):
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
        self._lattice = np.copy(lattice)
        self._init_pos = np.copy(init_pos)
        
    def extract_void_coordinates(self, new_positions):
        voids = {}
        coords2unwrap = np.append(self._init_pos,new_positions,axis=0)
        unw = unwrap_PBC(coords2unwrap,self._init_pos.shape[0])
        new_coords_unw = unw[self._init_pos.shape[0]:,:]
        for key in self.voids.keys():
            direct_coord_void = self.voids[key]['TetraVoidSC_PBC']
            disps = self.voids[key]['vertex_disp']
            indeces = self.voids[key]['indeces']
            void_direct_coords = new_coords_unw[indeces,:].reshape(direct_coord_void.shape[0],3) + disps 
            voids[key] = frac2cart(void_direct_coords,self._lattice)
        return voids
