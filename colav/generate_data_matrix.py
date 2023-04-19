
import numpy as np 

def extract_off_diagonals(shear_tensors): 
    '''
    Find the off diagonal elements of the shear tensors
    '''
    
    # initialize array for storage of shear values 
    off_diags = list()
    
    # iterate through each structure 
    for i in range(shear_tensors.shape[0]): 
        
        # extend the off diagonals by the off diagonal values for the current structure 
        off_diags.extend([tensor.flatten()[[1,2,5]] for tensor in shear_tensors[i,:,:,:]])
        
    return np.array(off_diags).reshape((shear_tensors.shape[0], shear_tensors.shape[1] * 3))

def generate_dihedral_matrix(structure_list: list, resnum_bounds, verbose=False): 

    '''
    Calculate and store the dihedral angles of a set of structures in 
    structure_list
    '''

    # set atoms for dihedral angle calculation
    atoms = ["N", "CA", "C"]

    # set of shared dihedral angles for each structure 
    dhset = list()
    dh_strucs = list()

    # iterate through the structural models
    print("Calculating the dihedral angles...")
    for i,struc in enumerate(structure_list): 

        # parse and clean the pdb files
        df = model_to_df(struc)

        # get the desired data 
        df = get_data(df, resnum_bounds=resnum_bounds, atoms=atoms)

        # check if the resulting sequence is consecutive and skip if not 
        # print(np.unique(df.resnum.values).shape[0] == (resnum_bounds[1] - resnum_bounds[0] + 1))
        if not np.unique(df.resnum.values).shape[0] == (resnum_bounds[1] - resnum_bounds[0] + 1): 
            
            continue

        # calculate and store list of dihedral angles for all structures
        dhset.append(np.array(calculate_phiomgpsis(df)))
        dh_strucs.append(struc)

    return np.array(dhset).astype("float64"), dh_strucs

def calculate_bb_dihedrals(structure_list: list, resnum_bounds, verbose=False): 
    '''
    Calculate and store all backbone dihedral angles (theta, phi, omega, psi) for a set of 
    structures in structure_list
    '''

    # set atoms for dihedral angle calculation 
    atoms = ["C", "CA", "N", "O"]

    # set of shared dihedral angles for each structure 
    dhset = list()
    dh_strucs = list()

    # iterate through the structural models
    print("Calculating the dihedral angles...")
    for i,struc in enumerate(structure_list): 

        # parse and clean the pdb files
        df = model_to_df(struc)

        # get the desired data 
        df = get_data(df, resnum_bounds=resnum_bounds, atoms=atoms)

        # check if the resulting sequence is consecutive and skip if not 
        if not np.unique(df.resnum.values).shape[0] == (resnum_bounds[1] - resnum_bounds[0] + 1): 
            
            continue

        # calculate and store list of dihedral angles for all structures
        dhset.append(np.array(calculate_dihedrals(df)))
        dh_strucs.append(struc)

    return np.array(dhset).astype("float64"), dh_strucs

def generate_pw_matrix(structure_list: list, resnum_bounds): 

    '''
    Calculate and store all the pairwise distances of a set of structures 
    in structure_list
    '''

    # set of coordinates for all structures 
    print("Generating the coordinate set...")
    coordset = generate_coord_matrix(structure_list=structure_list, resnum_bounds=resnum_bounds, atoms=["CA"])

    # initialize an array to store the pairwise distances 
    pw_dist = list()
    pw_strucs = list()

    # iterate through the structure coordinates
    print("Calculating the pairwise distances...")
    for i in range(coordset.shape[0]): 

        # check if there are NaNs in the coordinates 
        if np.any(np.isnan(coordset[i,:,:].flatten())): 

            continue

        # calculate the pairwise distances between all coordinates
        pw_dist.append(pdist(coordset[i,:,:]))

        pw_strucs.append(structure_list[i])

    return np.array(pw_dist).reshape(len(pw_strucs), -1)

def calculate_strain_dict(structure_list: list, ref_pdb, resnum_bounds, atoms=["N", "C", "CA", "CB", "O"], save=False, verbose=False): 

    '''
    Calculate the strain compared to a reference structure for the structures
    in structure_list
    This method returns a dictionary with all the strain analysis results
    '''

    # load the reference structure information 
    print("Loading the reference structure...")
    ref = model_to_df(ref_pdb)
    ref = get_data(ref, resnum_bounds=resnum_bounds, atoms=atoms)

    # determine the shared atom set for all the structures 
    print("Determining shared atom set...")

    # initialize set to find all shared atoms based on reference structure
    atom_set = set([tuple(x) for x in ref[["resnum", "atom_id"]].values.tolist()])

    # initialize array to store structures that are amenable to strain analysis
    filtered_strucs = list()
    failed_list = list()

    # iterate through all the structural models 
    for i,struc in enumerate(structure_list): 

        # parse and filter the structural data
        df = model_to_df(struc)
        df = get_data(df, resnum_bounds=resnum_bounds, atoms=atoms)

        # compare the atom set to the current model 
        current_atom_set = set([tuple(x) for x in df[["resnum", "atom_id"]].values.tolist()])

        # check if the sequence is consecutive 
        if sorted(current_atom_set)[0][0] + len(np.unique(np.array(sorted(current_atom_set))[:,0])) - 1 == sorted(current_atom_set)[-1][0]: 

            # append the amenable structure to the filtered list
            filtered_strucs.append(struc)
            atom_set = atom_set.intersection(current_atom_set)
        
        else: 

            # append the failed structure to the failed list 
            failed_list.append(struc)

    # save the atom set information if requested
    if save: 

        print("Saving the atom_set data!")
        pickle.dump(atom_set, open("atom_set.pkl", "wb"))
    
    # report the structures that are not amenable to strain analysis
    for struc in failed_list: 

        print(f"{struc} is not amenable to strain analysis!")

    # get the reference atom set 
    ref_atom_set = set([tuple(x) for x in ref[["resnum", "atom_id"]].values.tolist()])

    # create a dictionary to store the strain analysis data
    strain_dict = dict()

    print(f"Calculating strain against reference {os.path.split(ref_pdb)[1]}...")
    # calculate and store the strain information for the filtered structures 
    for i,struc in enumerate(filtered_strucs): 

        # skip if the reference structure 
        if ref_pdb == struc: 

            continue

        # report current structure
        if verbose: 
            
            print(f"{i}: Calculating for {os.path.split(struc)[1]}")

        # parse and then clean up the data 
        df = model_to_df(struc)
        df = get_data(df, atoms=atoms)

        # find the shared atoms 
        current_atom_set = set([tuple(x) for x in df[["resnum", "atom_id"]].values.tolist()])
        comp_atom_set = ref_atom_set.intersection(current_atom_set)

        # get the reference coordinates 
        ref_strain_coords = match_list(ref, sorted(comp_atom_set))[["x", "y", "z"]].to_numpy().astype("float64")

        # find the appropriate atoms for comparison and then calculate strain 
        current_strain_coords = match_list(df, sorted(comp_atom_set))[["x", "y", "z"]].to_numpy().astype("float64")
        sheare, sheart, straint = calculate_strain(ref_strain_coords, current_strain_coords)

        # find the idxs of the full atom set for the comparison atom set
        atom_idxs = list()
        comp_atom_list = sorted(comp_atom_set)
        comp_idx = 0
        
        # sort out the B-factors 
        bfacs = match_list(df, sorted(comp_atom_set))["bfac"].values.astype("float64")
        if np.any(bfacs == 0): 
            print(f"Skipping {struc} because B factor of 0!")
            continue

        # iterate through the list and append the relevant atom indices to atom_idxs
        for i,atom in enumerate(sorted(atom_set)): 

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

        print("Saving the strain_dict data!")
        pickle.dump(strain_dict, open("strain_dict.pkl", "wb"))

    return strain_dict, atom_set

def generate_strain_matrix(structure_list: list, ref_pdb, data_type, resnum_bounds, atoms=["N", "C", "CA", "CB", "O"], save=False): 

    '''
    Returns a strain analysis matrix calculated for the structures 
    against the reference
    '''

    # check if there's already an existing pkl file 
    if "strain_dict.pkl" in os.listdir() and "atom_set.pkl" in os.listdir(): 

        strain_dict = pickle.load(open("strain_dict.pkl", "rb"))
        atom_set = pickle.load(open("atom_set.pkl", "rb"))

    # if not then calculate the strain dictionary 
    else: 

        strain_dict, atom_set = calculate_strain_dict(
            structure_list=structure_list, 
            ref_pdb=ref_pdb, 
            resnum_bounds=resnum_bounds, 
            atoms=atoms, 
            save=True
        )

    # generate the data matrix 
    data_matrix = list()

    print(f"Generating desired {data_type} data matrix...")
    # iterate through the keys of the shear dictionary and filtered structures 
    for key in strain_dict.keys():

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
            ValueError

        # grab the relevant data based on data_type and apply the correction 
        if "P0125_0" in key: 
            for i in range(len(atom_set)): 
                print(f"{np.square(bfacs[i])}")

        processed /= correction 

        # store the data
        data_matrix.append(processed)

    return np.array(data_matrix), atom_set

def load_strain_matrix(strain_pkl, atom_pkl, data_type): 
    '''
    Load the relevant data 
    '''
        
    strain_dict = pickle.load(open(f"{strain_pkl}", "rb"))
    atom_set = pickle.load(open(f"{atom_pkl}", "rb"))

    # generate the data matrix 
    data_matrix = list()

    print(f"Generating desired {data_type} data matrix...")
    # iterate through the keys of the shear dictionary and filtered structures 
    for key in strain_dict.keys():

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
            ValueError

        # grab the relevant data based on data_type and apply the correction 
        if "P0125_0" in key: 
            for i in range(len(atom_set)): 
                print(f"{np.square(bfacs[i])}")

        processed /= correction 

        # store the data
        data_matrix.append(processed)

    return np.array(data_matrix), atom_set
