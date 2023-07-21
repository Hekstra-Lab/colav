
import os, pickle
import numpy as np 
from biopandas.pdb import PandasPdb
from itertools import combinations
from colav.strain_analysis import *
from colav.internal_coordinates import * 
from scipy.spatial.distance import pdist

def calculate_dh_tl(raw_dh_loading): 
    '''Adjusts raw dihedral loading for interpretability. 

    Calculates a transformed loading from a raw loading of dihedral angle features to account 
    for the application of sine and cosine functions. 

    Returns a transformed loading. 

    Parameters: 
    -----------
    raw_dh_loading : array_like, (N,)
        Array of raw loading from PCA. 

    Returns: 
    --------
    tranformed_dh_loading : array_like, (N/2,)
        Array of transformed loading to determine relative angle influence in the given 
        loading. 
    '''
    
    raw_dh_loading = np.array(raw_dh_loading)
    tranformed_dh_loading = np.abs(raw_dh_loading[:raw_dh_loading.shape[0]//2]) + np.abs(raw_dh_loading[raw_dh_loading.shape[0]//2:])
    return tranformed_dh_loading

def generate_dihedral_matrix(structure_list, resnum_bounds, no_psi=False, no_omega=False, no_phi=False, save=False, save_prefix=None, verbose=False): 
    '''Extracts dihedrals angles from given structures. 

    Extracts and returns a data matrix of (observations x features) with the given structures as observations
    and the linearized dihedral angles (by applying sine and cosine functions) as features. 
    Cannot handle missing coordinates and skips structures with missing backbone atoms within the 
    given residue numbers. 

    Parameters: 
    -----------
    structure_list : list of str
        Array containing the file paths to PDB structures. 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values. 

    no_psi : bool, optional 
        Indicator to exclude psi dihedral angle from returned dihedral angles. 

    no_omega : bool, optional 
        Indicator to exclude omega dihedral angle from returned dihedral angles. 

    no_phi : bool, optional 
        Indicator to exclude phi dihedral angle from returned dihedral angles. 

    save : bool, optional 
        Indicator to save results. 

    save_prefix : str 
        If saving results, prefix for pickle save file. 

    verbose : bool, optional 
        Indicator for verbose output. 

    Returns: 
    --------
    dh_data_matrix : array_like 
        Array containing dihedral angles within `resnum_bounds` for structures in `structure_list`,  
        excluding structures missing desired atoms. 
    
    dh_strucs : list of str
        List of structures ordered as stored in `dh_data_matrix`. 
    '''

    # set of shared dihedral angles for each structure 
    raw_dihedrals = list()
    dihedral_strucs = list()

    # iterate through the structural models
    if verbose: 
        print("Calculating the dihedral angles...")

    for i,struc in enumerate(structure_list): 

        # parse the pdb files
        if verbose: 
            print(f"Attempting to calculate for {struc}")
        ppdb = PandasPdb().read_pdb(struc)
        mainchain = ppdb.df['ATOM'].loc[(ppdb.df['ATOM']['atom_name'] == 'N') | # choose the correct atoms 
                                        (ppdb.df['ATOM']['atom_name'] == 'CA')|
                                        (ppdb.df['ATOM']['atom_name'] == 'C')]
        mainchain = mainchain.loc[(mainchain['residue_number'] >= resnum_bounds[0]) & # choose the correct residue numbers 
                                  (mainchain['residue_number'] <= resnum_bounds[1])]
        mainchain = mainchain.loc[(mainchain['alt_loc'] == '') |  # choose the A alt_loc if there are any 
                                  (mainchain['alt_loc'] == 'A')]
        if np.unique(mainchain.residue_number.values).shape[0] != (resnum_bounds[1] - resnum_bounds[0] + 1): 
            if verbose: 
                print(f"Skipping {struc}; insufficient atoms!")
            continue
        
        dihedrals = calculate_backbone_dihedrals(
            ppdb=ppdb, 
            resnum_bounds=resnum_bounds, 
            no_psi=no_psi, 
            no_omega=no_omega, 
            no_phi=no_phi, 
            verbose=verbose
        )

        raw_dihedrals.append(dihedrals)
        dihedral_strucs.append(struc)

    raw_dihedrals = np.array(raw_dihedrals).reshape(len(dihedral_strucs), -1)
    
    # save the results of the calculation as a np array if desired 
    if save: 

        if verbose: 
            print('Saving dh_dict data!')
        # create a dictionary to store the data matrix and structures 
        dh_dict = {
            'data_matrix': raw_dihedrals, 
            'structures': dihedral_strucs
        }

        # save with prefix if it is given
        if save_prefix is None: 
            with open(f'dh_dict.pkl', 'wb') as f: 
                pickle.dump(dh_dict, f)

        else: 
            with open(f'{save_prefix}_dh_dict.pkl', 'wb') as f: 
                pickle.dump(dh_dict, f)

    return raw_dihedrals, dihedral_strucs

def load_dihedral_matrix(dh_pkl): 
    '''Loads the dihedral data matrix and corresponding structures 

    Loads a dictionary containing the dihedral data matrix as `data_matrix` key 
    and the corresponding structures as `structures` key

    Parameters: 
    -----------
    dh_pkl : str
        File path to the dihedral dictionary pickle file.
    
    Returns: 
    --------
    dh_data_matrix : array_like 
        Array containing dihedral angles as calculated by `generate_dihedral_matrix`. 
    
    dh_strucs : list of str
        List of structures ordered as stored in `dh_data_matrix`.
    '''

    # load the dictionary information 
    db = pickle.load(open(f'{dh_pkl}', 'rb'))
    dh_data_matrix = db['data_matrix']
    dh_strucs = db['structures']

    return dh_data_matrix, dh_strucs

def calculate_pw_tl(raw_pw_loading, resnum_bounds): 
    '''Adjusts raw pairwise distance loading for interpretability. 

    Calculates a transformed loading from a raw loading of pairwise distance features to account 
    for all pairings of residues. 

    Returns a transformed loading. 

    Parameters: 
    -----------
    raw_pw_loading : array_like, (N,)
        Array of raw loading from PCA. 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values. 

    Returns: 
    --------
    tranformed_pw_loading : array_like, (N/2,)
        Array of transformed loading to determine relative residue influence in the given loading. 
    '''
    
    raw_pw_loading = np.array(raw_pw_loading)
    # initialize array to store the contributions 
    tranformed_pw_loading = np.zeros(resnum_bounds[1]-resnum_bounds[0]+1)
    
    # create array of residue combos 
    pw_combos = np.array(list(combinations(np.arange(resnum_bounds[0], resnum_bounds[1]+1), 2)))
    
    # iterate through the pairs and store contributions in both (since order does not matter for contributions)
    for i,combo in enumerate(pw_combos): 
        
        # access the residues and add contributions for both contributors 
        tranformed_pw_loading[combo[0]-resnum_bounds[0]] += np.abs(raw_pw_loading[i])
        tranformed_pw_loading[combo[1]-resnum_bounds[0]] += np.abs(raw_pw_loading[i])
        
    return tranformed_pw_loading

def generate_pw_matrix(structure_list, resnum_bounds, save=False, save_prefix=None, verbose=False): 
    '''Extracts pairwise distances from given structures. 

    Extracts and returns a data matrix of (observations x features) with the given structures as observations
    and the pairwise distances between alpha carbon (CA) atoms as features. Cannot handle missing
    coordinates and skips structures with missing CA atoms within the given residue numbers. 
    
    Parameters:
    -----------
    structure_list : list of str
        Array containing the file paths to PDB structures. 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values. 

    save : bool, optional 
        Indicator to save results. 

    save_prefix : str
        If saving results, prefix for pickle save file. 

    verbose : bool, optional 
        Indicator for verbose output. 

    Returns: 
    --------
    pw_data_matrix : array_like 
        array containing pairwise distances between desired CA atoms for structures 
        in `structure_list`, excluding structures missing desired atoms. 

    pw_strucs : list of str
        List of structures ordered as stored in `pw_data_matrix`. 
    '''

    # initialize an array to store the pairwise distances and structures 
    pw_dist = list()
    pw_strucs = list()

    # set of coordinates for all structures 
    if verbose: 
        print("Generating the coordinate set...")
    for i,struc in enumerate(structure_list): 

        # parse the pdb files 
        if verbose: 
            print(f'Attempting to calculate for {struc}')
        ppdb = PandasPdb().read_pdb(struc)
        cas = ppdb.df['ATOM'][(ppdb.df['ATOM']['atom_name'] == 'CA')] # choose the correct atoms 
        cas = cas.loc[(ppdb.df['ATOM']['residue_number'] >= resnum_bounds[0]) & # choose the correct residue numbers 
                      (ppdb.df['ATOM']['residue_number'] <= resnum_bounds[1])]
        cas = cas.loc[(ppdb.df['ATOM']['alt_loc'] == '') |  # choose the A alt_loc if there are any 
                      (ppdb.df['ATOM']['alt_loc'] == 'A')]
        cas = cas.reset_index()

        # check that all pairs of CA atoms are present 
        if cas.shape[0] != (resnum_bounds[1] - resnum_bounds[0] + 1): 
            if verbose: 
                print(f'Skipping {struc}; not all desired CA atoms present!')
            continue
        
        # retrieve the CA coordinate information and calculate pairwise distances
        pw_strucs.append(struc)
        pw_dist.append(pdist(cas[['x_coord', 'y_coord', 'z_coord']].to_numpy()))

    pw_data_matrix = np.array(pw_dist).reshape(len(pw_strucs), -1)

    # save the results of the calculation as a np array if desired 
    if save: 

        if verbose: 
            print('Saving the pw_dict data!')

        # create a dictionary to store the data matrix and structures 
        pw_dict = {
            'data_matrix': pw_data_matrix, 
            'structures': pw_strucs
        }

        # save with prefix if it is given
        if save_prefix is None: 
            with open(f'pw_dict.pkl', 'wb') as f: 
                pickle.dump(pw_dict, f)

        else: 
            with open(f'{save_prefix}_pw_dict.pkl', 'wb') as f: 
                pickle.dump(pw_dict, f)

    return pw_data_matrix, pw_strucs

def load_pw_matrix(pw_pkl): 
    '''Loads the pairwise distance data matrix and corresponding structures 

    Loads a dictionary containing the pairwise distance data matrix as `data_matrix` key 
    and the corresponding structures as `structures` key

    Parameters: 
    -----------
    pw_pkl : str
        File path to the pairwise distance dictionary pickle file.
    
    Returns: 
    --------
    pw_data_matrix : array_like 
        Array containing dihedral angles as calculated by `generate_pw_matrix`. 
    
    pw_strucs : list of str
        List of structures ordered as stored in `pw_data_matrix`.
    '''

    # load the dictionary information 
    db = pickle.load(open(f'{pw_pkl}', 'rb'))
    pw_data_matrix = db['data_matrix']
    pw_strucs = db['structures']

    return pw_data_matrix, pw_strucs

def calculate_sa_tl(raw_sa_loading, shared_atom_list): 
    '''Adjusts raw strain or shear loading for interpretability. 

    Calculates a transformed loading from a raw loading of strain or shear tensor features. 

    Returns a transformed loading. 

    Parameters: 
    -----------
    raw_sa_loading : array_like
        Array of raw loading from PCA. 

    shared_atom_list : array_like 
        Sorted list of shared atoms between all structures used for strain analysis. 

    Returns: 
    --------
    tranformed_sa_loading : array_like 
        Array of transformed loading to determine relative residue influence in the given loading. 
    '''
    
    raw_sa_loading = np.array(raw_sa_loading)
    # first find atomic contributions 
    atomic_contributions = np.sum(np.abs(raw_sa_loading.reshape(-1,3)), axis=1)
    
    # create list of resnums 
    shared_atom_list = np.array(shared_atom_list)
    resnum_list = shared_atom_list[:,0].astype("int64")

    # ensure that the number of atoms is consistent
    assert(resnum_list.shape[0] == atomic_contributions.shape[0])
    
    # find unique residue numbers 
    unq_resnums = np.unique(resnum_list)
    
    # initialize array to store the contributions
    tranformed_sa_loading = np.zeros(unq_resnums.shape)
    
    # iterate through residue numbers 
    for i,resnum in enumerate(unq_resnums): 
        
        # access the contributions and sum 
        tranformed_sa_loading[i] += np.sum(atomic_contributions[resnum_list == resnum])
        
    return tranformed_sa_loading

def generate_strain_matrix(structure_list, reference_pdb, data_type, resnum_bounds, atoms=["N", "C", "CA", "CB", "O"], alt_locs=["", "A"], save=True, save_prefix=None, verbose=False): 
    '''Extracts strain tensors, shear tensors, or shear energies from given structures. 

    Extracts and returns a data matrix of (observations x features) with the given structures as observations
    and strain tensors, shear tensors, or shear energies. For tensor features, only the off-diagonal 
    elements are included. Cannot handle missing coordinates and skips structures with missing 
    backbone atoms within the given residue numbers. 
    
    Parameters:
    -----------
    structure_list : list of str 
        Array containing the file paths to PDB structures. 

    reference_pdb : str
        File path to the reference PDB structure; this structure can be contained in `structure_list`. 

    data_type : {'straint', 'sheart', 'sheare'}
        Indicator for type of data to build data matrix. 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values. 

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
    sa_data_matrix : array_like 
        Array containing strain or shear tensor information structures in `structure_list`, 
        excluding structures missing desired atoms. 

    sa_strucs : list of str
        List of structures ordered as stored in the `sa_data_matrix`. 
    '''

    # calculate a strain dictionary 
    if verbose: 
        print("Creating a strain dictionary. Calculating...")
    strain_dict, atom_set = calculate_strain_dict(
        structure_list=structure_list, 
        reference=reference_pdb, 
        resnum_bounds=resnum_bounds, 
        atoms=atoms, 
        alt_locs=alt_locs,
        save=False, 
        save_prefix=save_prefix, 
        verbose=verbose
    )

    # generate the data matrix 
    sa_data_matrix = list()
    sa_strucs = list()

    if verbose: 
        print(f"Generating desired {data_type} data matrix...")
    # iterate through the keys of the shear dictionary and filtered structures 
    for key in sorted(strain_dict.keys()):

        if verbose: 
            print(f"Attempting to calculate {data_type} matrix for {key}")
        # access the data
        atom_data = strain_dict[key][data_type][strain_dict[key]["atom_idxs"]]

        # get the B-factors and shape the correction scale
        bfacs = np.sqrt(strain_dict[key]["bfacs"][strain_dict[key]["atom_idxs"]])

        # choose the correction to match the strain/shear data selected for analysis
        if data_type == "sheart" or data_type == "straint": 
            correction = np.hstack([bfacs[:,None], bfacs[:,None], bfacs[:,None]]).flatten()
            processed = np.array([tensor[np.triu_indices(3,1)] for tensor in atom_data]).flatten() # off diagonals

        elif data_type == "sheare":
            correction = bfacs
            processed = np.array(atom_data)

        else: 
            ValueError("Must be sheare, sheart, or straint")

        # apply the coorection and store the data
        processed = processed / correction 
        sa_data_matrix.append(processed)
        sa_strucs.append(key)

    # save the results of the calculation as a np array if desired 
    if save: 

        if verbose: 
            print('Saving the sa_dict data!')

        # create a dictionary to store the data matrix and structures 
        sa_dict = {
            'data_matrix': np.array(sa_data_matrix), 
            'structures': sa_strucs
        }

        # save with prefix if it is given
        if save_prefix is None: 
            with open(f'sa_dict.pkl', 'wb') as f: 
                pickle.dump(sa_dict, f)

        else: 
            with open(f'{save_prefix}_sa_{data_type}_dict.pkl', 'wb') as f: 
                pickle.dump(sa_dict, f)

    return np.array(sa_data_matrix), sa_strucs

def load_strain_matrix(strain_pkl): 
    '''Loads the strain data matrix and corresponding structures 

    Loads a dictionary containing the strain data matrix as `data_matrix` key 
    and the corresponding structures as `structures` key

    Parameters: 
    -----------
    strain_pkl : str
        File path to the strain dictionary pickle file.
    
    Returns: 
    --------
    sa_data_matrix : array_like 
        Array containing dihedral angles as calculated by `generate_sa_matrix`. 
    
    sa_strucs : list of str
        List of structures ordered as stored in `sa_data_matrix`.
    '''

    # load the dictionary information 
    db = pickle.load(open(f'{strain_pkl}', 'rb'))
    sa_data_matrix = db['data_matrix']
    sa_strucs = db['structures']

    return np.array(sa_data_matrix), sa_strucs