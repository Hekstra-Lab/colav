
import pickle
import numpy as np 
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

def calculate_strain(ref_coords, def_coords, min_dist=6, max_dist=8, err_threshold=10, verbose=False): 
    '''Calculates strain quantities for a perturbed structure with respect to a reference. 

    Calculates the positional shear energy, positional shear tensor, and positional 
    shear tensor for the provided atoms by comparing the perturbed/deformed 
    structure (`def_coords`) to the reference structure (`ref_coords`). Note, 
    the atomic neighborhood of each atom is defined by a spherical shell 
    with radius 8 Angstroms. 

    Returns a shear energy, shear tensor, and strain tensor for each coordinate (i.e., 
    atom) in `def_coords`. 

    This code was heavily inspired by K. Ian White's implementation of strain 
    calculations in `delta_r_analysis.py` (not published).
    
    Parameters: 
    -----------
    ref_coords : array_like, (N x 3)
        Cartesian coordinates (x, y, z) of atoms in the reference structure. 

    def_coords : array_like, (N x 3)
        Cartesian coordinates (x, y, z) of atoms in the deformed structure. 

    min_dist : int 
        Minimum distance to apply atomic neighborhood weighting scheme. 
    
    max_dist : int
        Maximum distance to apply atomic neighborhood weighting scheme. 

    err_threshold : int 
        Threshold to ensure that the shear energies are not too large and potentially 
        reflecting errors in calculation (e.g., due to non-invertible matrices). 

    verbose : bool, optional 
        Indicator for verbose output. 

    Returns: 
    --------
    pos_shear_energy : array_like, (N,)
        Array of shear energies for each atom. 

    pos_shear_tensor : array_like, (N x 3 x 3)
        Array of shear tensors for each atom. 

    pos_strain_tensor : array_like, (N x 3 x 3)
        Array of strain tensors for each atom. 
    '''
    
    # ensure that length of arrays are the same (proxy for checking individual atoms)
    assert(ref_coords.shape == def_coords.shape)
    
    # initialize arrays for storage
    n_atoms = ref_coords.shape[0]
    pos_shear_energy = np.empty(n_atoms) 
    pos_shear_tensor = np.empty([n_atoms,3,3])
    pos_strain_tensor = np.empty([n_atoms,3,3])
    
    # calculate distance matrices for protein models from coordinates
    ref_dist = cdist(ref_coords, ref_coords)
    
    # iterate through the atoms being considered 
    for atom_idx in range(n_atoms): 

        if verbose: 
            print(f'Working through atom number: {atom_idx+1}!')

        # use counters 
        it_num = -1
        err_num = 0
        while err_num != it_num: 

            if verbose: 
                print(f'Iteration number: {it_num}\tError number: {err_num}')

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

            if verbose: 
                print(f'pos_strain_tensor: {Em}\tpos_shear_tensor: {gamma}\tpos_shear_energy: {np.sum(np.square(gamma))}')

            # update the counters 
            it_num += 1
            if pos_shear_energy[atom_idx] > err_threshold: 
                err_num += 1
        
    return pos_shear_energy, pos_shear_tensor, pos_strain_tensor

def determine_shared_atoms(structure_list, reference_ppdb, resnum_bounds, atoms=["N", "C", "CA", "CB", "O"], alt_locs=["", "A"], save=False, save_prefix=None, verbose=False): 
    '''Determines shared atoms across structures. 
    
    Determines the shared atoms for all structures in a list and a given 
    reference structure between given residue numbers (`resnum_bounds`) and for 
    explicitly desired atoms (`atoms`). 

    Returns the shared set of atoms for all given structures. 

    Parameters: 
    -----------
    structure_list : list of str
        Array containing the file paths to PDB structures. 

    reference_ppdb : PandasPdb (ATOM header only)
        Dataframe containing ATOM information of protein structure; 
        e.g., ppdb.df['ATOM']; this structure can be contained in `structure_list`. 

    resnum_bounds : tuple 
        Tuple containing the minimum and maximum (inclusive) residue number values. 

    atoms : array_like, optional 
        Array containing atom names. 

    alt_locs : array_like, optional 
        Array containing alternate location names. 

    save : bool, optional 
        Indicator to save the shared atom set. 

    save_prefix : str 
        If saving results, prefix for pickle save file. 

    verbose : bool, optional 
        Indicator for verbose output. 

    Returns: 
    --------
    ref_atom_set : set 
        Set containing shared atoms for all structures in `structure_list` 
        and `reference_ppdb`.

    filtered_strucs : list 
        Array containing the file paths to PDB structures that contain all shared 
        atoms. If the original `structure_list` contains the reference protein structure, 
        then it will be included here. 
    '''

    # determine the shared atom set for all the structures 
    if verbose: 
        print("Determining shared atom set...")

    # initialize set to find all shared atoms based on reference structure
    atom_set = set([tuple(x) for x in reference_ppdb[['residue_number', 'atom_name']].values.tolist()])

    # initialize array to store structures that are amenable to strain analysis
    filtered_strucs = list()
    failed_list = list()

    # iterate through all the structural models 
    for i,struc in enumerate(structure_list): 

        if verbose: 
            print(f"Working through structure {struc}")
        # parse and filter the structural data
        ppdb = PandasPdb().read_pdb(struc)
        use_atoms = np.zeros(ppdb.df['ATOM'].shape[0])
        use_locs = np.zeros(ppdb.df['ATOM'].shape[0])
        for i,atom in enumerate(atoms): 
            use_atoms = use_atoms | (ppdb.df['ATOM']['atom_name'] == atom)
        for i,loc in enumerate(alt_locs): 
            use_locs = use_locs | (ppdb.df['ATOM']['alt_loc'] == loc)
        df = ppdb.df['ATOM'][(use_atoms) & 
                             (use_locs) & 
                             (ppdb.df['ATOM']['residue_number'] >= resnum_bounds[0]) &
                             (ppdb.df['ATOM']['residue_number'] <= resnum_bounds[1])]

        # compare the atom set to the current model 
        current_atom_set = set([tuple(x) for x in df[['residue_number', 'atom_name']].values.tolist()])

        # check if the sequence is completely intact and consecutive 
        if sorted(current_atom_set)[0][0] + len(np.unique(np.array(sorted(current_atom_set))[:,0])) - 1 == sorted(current_atom_set)[-1][0]: 

            # append the amenable structure to the filtered list
            filtered_strucs.append(struc)
            atom_set = atom_set.intersection(current_atom_set)
        
        else: 

            # append the failed structure to the failed list 
            failed_list.append(struc)

    # save the atom set information if requested
    if save: 
        if verbose: 
            print("Saving the atom_set data!")
        
        if save_prefix is None: 
            with open(f'atom_set.pkl', 'wb') as f: 
                pickle.dump(atom_set, f)
        else: 
            with open(f'{save_prefix}_atom_set.pkl', 'wb') as f: 
                pickle.dump(atom_set, f)
    
    # report the structures that are not amenable to strain analysis
    for struc in failed_list: 

        if verbose: 
            print(f"{struc} does not share the necessary atoms!")

    # get the reference atom set 
    ref_atom_set = atom_set.intersection(set([tuple(x) for x in reference_ppdb[['residue_number', 'atom_name']].values.tolist()]))

    return ref_atom_set, filtered_strucs

def coords_from_atoms(struc_df, sorted_atom_list): 
    '''Retrieves and returns the Cartesian coordinates of the supplied strucure for 
    the given atoms. 

    Parameters: 
    -----------
    struc_df : PandasPdb (ATOM header only)
        Dataframe containing ATOM information of protein structure; 
        e.g., ppdb.df['ATOM']. 

    sorted_atom_list : array_like
        Sorted list containing atoms (residue number and atom name) for which
        to retrieve the desired coordinate information. 

    Returns: 
    --------
    xyz_coords : array_like, (N x 3)
        Array of Cartesian coordinates for the atoms of the given structure. 
    '''

    # initialize array for the coordinates 
    xyz_coords = list()

    # enumerate through all atoms 
    for i,atom in enumerate(sorted_atom_list): 
        xyz_coords.append(struc_df.loc[np.logical_and(struc_df.residue_number == atom[0], 
                                                    struc_df.atom_name == atom[1])] \
                                      [['x_coord', 'y_coord', 'z_coord']].to_numpy())
    
    return np.array(xyz_coords).reshape(-1,3)

def bfacs_from_atoms(struc_df, sorted_atom_list): 
    '''Retrieves and returns the B factors of the supplied strucure for 
    the given atoms. 

    Parameters: 
    -----------
    struc_df : PandasPdb (ATOM header only)
        Dataframe containing ATOM information of protein structure; 
        e.g., ppdb.df['ATOM']. 

    sorted_atom_list : array_like
        Sorted list containing atoms (residue number and atom name) for which
        to retrieve the desired coordinate information. 

    Returns: 
    --------
    bfacs : array_like 
        Array of B factors for the atoms of the given structure. 
    '''

    # initialize array for the coordinates 
    bfacs = list()

    # enumerate through all atoms 
    for i,atom in enumerate(sorted_atom_list): 
        bfacs.append(struc_df.loc[np.logical_and(struc_df.residue_number == atom[0], 
                                                    struc_df.atom_name == atom[1])].b_factor.to_numpy())
    
    return np.array(bfacs).reshape(-1)

def calculate_strain_dict(structure_list, reference, resnum_bounds, atoms=["N", "C", "CA", "CB", "O"], alt_locs=["", "A"], save=True, save_prefix=None, verbose=False): 
    '''Stores strain information in a dictionary for later ease of use. 

    Constructs a dictionary object containing strain and shear information for all given 
    structures in `structure_list` compared to the given reference structure within 
    the given residue numbers (`resnum_bounds`) and using only the explicitly desired 
    atoms (`atoms`). Note, ssumes that the numbering of residues in each of the structures 
    is consistent across the entire data set. 

    Parameters:
    -----------
    structure_list : list of str
        Array containing the file paths to PDB structures. 

    reference : string 
        File path to the reference PDB structure; this structure can be contained 
        in `structure_list`. 

    resnum_bounds : tuple 
        Tuple containing the minimum and maximum (inclusive) residue number values 
        for strain calculations. 

    atoms : array_like, optional 
        Array containing atom names. 

    alt_locs : array_like, optional 
        Array containing alternate location names. 

    save : bool, optional 
        Indicator to save results. 
    
    save_prefix : str 
        If saving results, prefix for pickle save file. 

    verbose : bool, optional 
        Indicator for verbose output. 

    Returns: 
    --------
    strain_dict : dictionary
        Dictionary containing results of strain calculations for the shared atoms of all 
        structures in `structure_list` compared to `reference`. 
    
    shared_atom_set : set 
        Set of shared atoms (determined by `determine_shared_atoms`). 
    '''

    # load the reference structure 
    if verbose: 
        print('Loading the reference structure...')
    ppdb = PandasPdb().read_pdb(reference)
    use_atoms = np.zeros(ppdb.df['ATOM'].shape[0])
    use_locs = np.zeros(ppdb.df['ATOM'].shape[0])
    for i,atom in enumerate(atoms): 
        use_atoms = use_atoms | (ppdb.df['ATOM']['atom_name'] == atom)
    for i,loc in enumerate(alt_locs): 
        use_locs = use_locs | (ppdb.df['ATOM']['alt_loc'] == loc)
    ref_ppdb = ppdb.df['ATOM'][(use_atoms) & 
                               (use_locs) & 
                               (ppdb.df['ATOM']['residue_number'] >= resnum_bounds[0]) &
                               (ppdb.df['ATOM']['residue_number'] <= resnum_bounds[1])]

    shared_atom_set, filtered_strucs = determine_shared_atoms(
        structure_list=structure_list, 
        reference_ppdb=ref_ppdb,
        resnum_bounds=resnum_bounds, 
        atoms=atoms, 
        alt_locs=alt_locs,
        save=save, 
        save_prefix=save_prefix,
        verbose=verbose
    )

    # determine the relevant coordinates for the reference structure
    ref_strain_coords = coords_from_atoms(ref_ppdb, sorted(shared_atom_set))

    # create a dictionary to store the strain analysis data
    strain_dict = dict()

    print(f"Calculating strain against reference {reference}...")
    # calculate and store the strain information for the filtered structures 
    for i,struc in enumerate(sorted(filtered_strucs)): 

        # skip if the reference structure 
        if reference == struc: 

            continue

        # report current structure
        if verbose: 
            print(f"Working through {struc} for strain calculations")

        # parse and then clean up the data 
        ppdb = PandasPdb().read_pdb(struc)
        use_atoms = np.zeros(ppdb.df['ATOM'].shape[0])
        use_locs = np.zeros(ppdb.df['ATOM'].shape[0])
        for i,atom in enumerate(atoms): 
            use_atoms = use_atoms | (ppdb.df['ATOM']['atom_name'] == atom)
        for i,loc in enumerate(alt_locs): 
            use_locs = use_locs | (ppdb.df['ATOM']['alt_loc'] == loc)
        df = ppdb.df['ATOM'][(use_atoms) & 
                             (use_locs) & 
                             (ppdb.df['ATOM']['residue_number'] >= resnum_bounds[0]) &
                             (ppdb.df['ATOM']['residue_number'] <= resnum_bounds[1])]
        

        # determine coordinates for strain calculations 
        current_atom_set = set([tuple(x) for x in df[['residue_number', 'atom_name']].values.tolist()])
        comp_atom_set = shared_atom_set.intersection(current_atom_set)
        current_strain_coords = coords_from_atoms(df, sorted(comp_atom_set))

        sheare, sheart, straint = calculate_strain(ref_strain_coords, current_strain_coords)

        # find the idxs of the full atom set for the comparison atom set
        atom_idxs = list()
        comp_atom_list = sorted(comp_atom_set)
        comp_idx = 0
        
        # sort out the B-factors 
        bfacs = bfacs_from_atoms(df, sorted(comp_atom_set))
        if np.any(bfacs == 0): 
            print(f"Skipping {struc} because B factor of 0!")
            continue

        # iterate through the list and append the relevant atom indices to atom_idxs
        for i,atom in enumerate(sorted(shared_atom_set)): 

            # compare the current comp atom to the atom in atom set 
            while comp_atom_list[comp_idx] < atom:

                # if the atom is less than atom in atom set, then increase the current atom
                comp_idx += 1
                
            # when the current comp atom is equivalent to the atom in atom set, add to idx
            atom_idxs.append(comp_idx)

        strain_dict[struc] = {
            "straint": straint, 
            "sheart": sheart, 
            "sheare": sheare, 
            "bfacs": bfacs,
            "atom_list": sorted(comp_atom_set),
            "atom_idxs": atom_idxs
        }
    
    # save the strain dictionary information if requested
    if save: 

        if verbose: 
            print("Saving the strain_dict data!")

        if save_prefix is None: 
            with open(f'strain_dict.pkl', 'wb') as f: 
                pickle.dump(strain_dict, f)
        else: 
            with open(f'{save_prefix}_strain_dict.pkl', 'wb') as f: 
                pickle.dump(strain_dict, f)

    return strain_dict, shared_atom_set