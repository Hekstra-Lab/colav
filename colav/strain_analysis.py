
import numpy as np 
from scipy.spatial.distance import cdist

def calculate_strain(ref_coords, def_coords, min_dist=6, max_dist=8, err_threshold=10): 
    '''
    Calculate the strain on protein with respect to a reference 
    Of course, heavily inspired by Ian White's `delta_r_analysis.py`
    '''
    
    # ensure that length of arrays are the same (proxy for checking individual atoms)
    assert(ref_coords.shape == def_coords.shape)
    
    # initialize arrays for storage
    n_atoms = ref_coords.shape[0]
    pos_shear_energy = np.empty(n_atoms) * np.nan
    pos_shear_tensor = np.empty([n_atoms,3,3]) * np.nan
    pos_strain_tensor = np.empty([n_atoms,3,3]) * np.nan
    
    # calculate distance matrices for protein models from coordinates
    ref_dist = cdist(ref_coords, ref_coords)
    # def_dist = cdist(def_coords, def_coords)
    
    # iterate through the atoms being considered 
    for atom_idx in range(n_atoms): 

        # use counters 
        it_num = -1
        err_num = 0
        while err_num != it_num: 

            # set atomic weight contributions by distance; nearest atoms are weighted 1
            wn = (ref_dist[:,atom_idx] < min_dist).astype('float64')
            
            # if they are within the specified range, then calculate the contribution 
            in_range = np.where((ref_dist[:,atom_idx] >= min_dist) & (ref_dist[:,atom_idx] <= max_dist+err_num))
            wn[in_range] = 1 - (0.5 * (ref_dist[in_range, atom_idx] - min_dist))
            
            # define storage tensors 
            Dm = np.zeros((3,3))
            Am = np.zeros((3,3))
            
            # iterate through the neighbors of the current atom (as per wn)
            for neighbor_idx in np.where(wn != 0)[0]: 
            
                # calculate vector distances 
                ref_d = (ref_coords[neighbor_idx,:] - ref_coords[atom_idx,:]).reshape((3,1)).astype('float64')
                def_d = (def_coords[neighbor_idx,:] - def_coords[atom_idx,:]).reshape((3,1)).astype('float64')
                
                # calculate storage tensors 
                Dm += ref_d @ ref_d.T * wn[neighbor_idx]
                Am += def_d @ ref_d.T * wn[neighbor_idx]
                
            # calculate the gradient 
            Fm = np.dot(Am, np.linalg.inv(Dm))
            
            # calculate the strain tensor and shear tensor 
            Em = 0.5 * (np.eye(3) - np.linalg.inv(np.dot(Fm, Fm.T))) # using Euler-Almansi
            gamma = Em - ((np.trace(Em)/3)*np.eye(3))
            
            # store the strain tensor and shear energy 
            pos_strain_tensor[atom_idx,:,:] = Em 
            pos_shear_tensor[atom_idx,:,:] = gamma
            pos_shear_energy[atom_idx] = np.sum(np.square(gamma))

            # update the counters 
            it_num += 1
            if pos_shear_energy[atom_idx] > err_threshold: 
                err_num += 1
        
    return pos_shear_energy, pos_shear_tensor, pos_strain_tensor