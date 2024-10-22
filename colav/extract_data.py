import pickle
import numpy as np
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
from itertools import combinations, product
from colav.strain_analysis import *
from colav.internal_coordinates import *
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from multiprocessing import Pool


def calculate_dh_rc(raw_dh_loading, quadrature=False):
    """Adjusts raw dihedral loading for interpretability.

    Calculates a residue contribution from a raw loading of dihedral angle features to account
    for the application of sine and cosine functions.

    Returns a residue contribution.

    Parameters:
    -----------
    raw_dh_loading : array_like, (N,)
        Array of raw loading from PCA.

    quadrature : bool, optional
        Indicator to calculate residue contributions in quadrature.

    Returns:
    --------
    tranformed_dh_loading : array_like, (N/2,)
        Array of residue contribution to determine relative angle influence in the given
        loading.
    """

    raw_dh_loading = np.array(raw_dh_loading)
    tranformed_dh_loading = np.abs(
        raw_dh_loading[: raw_dh_loading.shape[0] // 2]
    ) + np.abs(raw_dh_loading[raw_dh_loading.shape[0] // 2 :]) if not quadrature else np.sqrt(np.square(raw_dh_loading[: raw_dh_loading.shape[0] // 2]) + np.square(raw_dh_loading[raw_dh_loading.shape[0] // 2 :]))
    return tranformed_dh_loading


def generate_dihedral_matrix(
    structure_list,
    resnum_bounds,
    no_psi=False,
    no_omega=False,
    no_phi=False,
    save=False,
    save_prefix=None,
    verbose=False,
):
    """Extracts dihedrals angles from given structures.

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
    """

    # set of shared dihedral angles for each structure
    raw_dihedrals = list()
    dihedral_strucs = list()

    # iterate through the structural models
    if verbose:
        print("Calculating the dihedral angles...")

    for i, struc in enumerate(structure_list):
        # parse the structure files, either pdb or mmcif
        if verbose:
            print(f"Attempting to calculate for {struc}")
        pstructure = PandasPdb().read_pdb(struc) if struc.endswith(".pdb") else PandasMmcif().read_mmcif(struc)
        mainchain = pstructure.df["ATOM"].loc[
            (pstructure.df["ATOM"]["atom_name"] == "N")
            | (pstructure.df["ATOM"]["atom_name"] == "CA")  # choose the correct atoms
            | (pstructure.df["ATOM"]["atom_name"] == "C")
        ]
        mainchain = mainchain.loc[
            (mainchain["residue_number"] >= resnum_bounds[0])
            & (  # choose the correct residue numbers
                mainchain["residue_number"] <= resnum_bounds[1]
            )
        ]
        mainchain = mainchain.loc[
            (mainchain["alt_loc"] == "")
            | (mainchain["alt_loc"] == "A")  # choose the A alt_loc if there are any
        ]
        if np.unique(mainchain.residue_number.values).shape[0] != (
            resnum_bounds[1] - resnum_bounds[0] + 1
        ):
            if verbose:
                print(f"Skipping {struc}; insufficient atoms!")
            continue

        dihedrals = calculate_backbone_dihedrals(
            pstructure=pstructure,
            resnum_bounds=resnum_bounds,
            no_psi=no_psi,
            no_omega=no_omega,
            no_phi=no_phi,
            verbose=verbose,
        )

        raw_dihedrals.append(dihedrals)
        dihedral_strucs.append(struc)

    raw_dihedrals = np.array(raw_dihedrals).reshape(len(dihedral_strucs), -1)

    # save the results of the calculation as a np array if desired
    if save:
        if verbose:
            print("Saving dh_dict data!")
        # create a dictionary to store the data matrix and structures
        dh_dict = {"data_matrix": raw_dihedrals, "structures": dihedral_strucs}

        # save with prefix if it is given
        if save_prefix is None:
            with open(f"dh_dict.pkl", "wb") as f:
                pickle.dump(dh_dict, f)

        else:
            with open(f"{save_prefix}_dh_dict.pkl", "wb") as f:
                pickle.dump(dh_dict, f)

    return raw_dihedrals, dihedral_strucs


def load_dihedral_matrix(dh_pkl):
    """Loads the dihedral data matrix and corresponding structures

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
    """

    # load the dictionary information
    db = pickle.load(open(f"{dh_pkl}", "rb"))
    dh_data_matrix = db["data_matrix"]
    dh_strucs = db["structures"]

    return dh_data_matrix, dh_strucs


def calculate_pw_rc(raw_pw_loading, resnum_bounds):
    """Adjusts raw pairwise distance loading for interpretability.

    Calculates a residue contribution from a raw loading of pairwise distance features to account
    for all pairings of residues.

    Returns a residue contribution.

    Parameters:
    -----------
    raw_pw_loading : array_like, (N,)
        Array of raw loading from PCA.

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values.

    Returns:
    --------
    tranformed_pw_loading : array_like, (N/2,)
        Array of residue contribution to determine relative residue influence in the given loading.
    """

    raw_pw_loading = np.array(raw_pw_loading)
    # initialize array to store the contributions
    tranformed_pw_loading = np.zeros(resnum_bounds[1] - resnum_bounds[0] + 1)

    # create array of residue combos
    pw_combos = np.array(
        list(combinations(np.arange(resnum_bounds[0], resnum_bounds[1] + 1), 2))
    )

    # iterate through the pairs and store contributions in both (since order does not matter for contributions)
    for i, combo in enumerate(pw_combos):
        # access the residues and add contributions for both contributors
        tranformed_pw_loading[combo[0] - resnum_bounds[0]] += np.abs(raw_pw_loading[i])
        tranformed_pw_loading[combo[1] - resnum_bounds[0]] += np.abs(raw_pw_loading[i])

    return tranformed_pw_loading


def generate_pw_matrix(
    structure_list, resnum_bounds, save=False, save_prefix=None, verbose=False
):
    """Extracts pairwise distances from given structures.

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
    """

    # initialize an array to store the pairwise distances and structures
    pw_dist = list()
    pw_strucs = list()

    # set of coordinates for all structures
    if verbose:
        print("Generating the coordinate set...")
    for i, struc in enumerate(structure_list):
        # parse the structure files, either pdb or mmcif
        if verbose:
            print(f"Attempting to calculate for {struc}")
        pstructure = PandasPdb().read_pdb(struc) if struc.endswith('.pdb') else PandasMmcif().read_mmcif(struc)
        cas = pstructure.df["ATOM"][
            (pstructure.df["ATOM"]["atom_name"] == "CA")
        ]  # choose the correct atoms
        cas = cas.loc[
            (pstructure.df["ATOM"]["residue_number"] >= resnum_bounds[0])
            & (  # choose the correct residue numbers
                pstructure.df["ATOM"]["residue_number"] <= resnum_bounds[1]
            )
        ]
        cas = cas.loc[
            (pstructure.df["ATOM"]["alt_loc"] == "")
            | (  # choose the A alt_loc if there are any
                pstructure.df["ATOM"]["alt_loc"] == "A"
            )
        ]
        cas = cas.reset_index()

        # check that all pairs of CA atoms are present
        if cas.shape[0] != (resnum_bounds[1] - resnum_bounds[0] + 1):
            if verbose:
                print(f"Skipping {struc}; not all desired CA atoms present!")
            continue

        # retrieve the CA coordinate information and calculate pairwise distances
        pw_strucs.append(struc)
        pw_dist.append(pdist(cas[["x_coord", "y_coord", "z_coord"]].to_numpy()))

    pw_data_matrix = np.array(pw_dist).reshape(len(pw_strucs), -1)

    # save the results of the calculation as a np array if desired
    if save:
        if verbose:
            print("Saving the pw_dict data!")

        # create a dictionary to store the data matrix and structures
        pw_dict = {"data_matrix": pw_data_matrix, "structures": pw_strucs}

        # save with prefix if it is given
        if save_prefix is None:
            with open(f"pw_dict.pkl", "wb") as f:
                pickle.dump(pw_dict, f)

        else:
            with open(f"{save_prefix}_pw_dict.pkl", "wb") as f:
                pickle.dump(pw_dict, f)

    return pw_data_matrix, pw_strucs


def load_pw_matrix(pw_pkl):
    """Loads the pairwise distance data matrix and corresponding structures

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
    """

    # load the dictionary information
    db = pickle.load(open(f"{pw_pkl}", "rb"))
    pw_data_matrix = db["data_matrix"]
    pw_strucs = db["structures"]

    return pw_data_matrix, pw_strucs


def calculate_sa_rc(raw_sa_loading, shared_atom_list):
    """Adjusts raw strain or shear loading for interpretability.

    Calculates a residue contribution from a raw loading of strain or shear tensor features.

    Returns a residue contribution.

    Parameters:
    -----------
    raw_sa_loading : array_like
        Array of raw loading from PCA.

    shared_atom_list : array_like
        Sorted list of shared atoms between all structures used for strain analysis.

    Returns:
    --------
    tranformed_sa_loading : array_like
        Array of residue contribution to determine relative residue influence in the given loading.
    """

    raw_sa_loading = np.array(raw_sa_loading)
    # first find atomic contributions
    atomic_contributions = np.sum(np.abs(raw_sa_loading.reshape(-1, 3)), axis=1)

    # create list of resnums
    shared_atom_list = np.array(shared_atom_list)
    resnum_list = shared_atom_list[:, 0].astype("int64")

    # ensure that the number of atoms is consistent
    assert resnum_list.shape[0] == atomic_contributions.shape[0]

    # find unique residue numbers
    unq_resnums = np.unique(resnum_list)

    # initialize array to store the contributions
    tranformed_sa_loading = np.zeros(unq_resnums.shape)

    # iterate through residue numbers
    for i, resnum in enumerate(unq_resnums):
        # access the contributions and sum
        tranformed_sa_loading[i] += np.sum(atomic_contributions[resnum_list == resnum])

    return tranformed_sa_loading


def generate_strain_matrix(
    structure_list,
    reference_pdb,
    data_type,
    resnum_bounds,
    atoms=["N", "C", "CA", "CB", "O"],
    alt_locs=["", "A"],
    save=True,
    save_prefix=None,
    save_additional=False,
    verbose=False,
):
    """Extracts strain tensors, shear tensors, or shear energies from given structures.

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

    save_additional : bool, optional
        Indicator to save results of nested calculations.

    verbose : bool, optional
        Indicator for verbose output.

    Returns:
    --------
    sa_data_matrix : array_like
        Array containing strain or shear tensor information structures in `structure_list`,
        excluding structures missing desired atoms.

    sa_strucs : list of str
        List of structures ordered as stored in the `sa_data_matrix`.
    """

    # calculate a strain dictionary
    if verbose:
        print("Creating a strain dictionary. Calculating...")
    strain_dict, atom_set = calculate_strain_dict(
        structure_list=structure_list,
        reference=reference_pdb,
        resnum_bounds=resnum_bounds,
        atoms=atoms,
        alt_locs=alt_locs,
        save=save_additional,
        save_prefix=save_prefix,
        verbose=verbose,
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
            correction = np.hstack(
                [bfacs[:, None], bfacs[:, None], bfacs[:, None]]
            ).flatten()
            processed = np.array(
                [tensor[np.triu_indices(3, 1)] for tensor in atom_data]
            ).flatten()  # off diagonals

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
            print("Saving the sa_dict data!")

        # create a dictionary to store the data matrix and structures
        sa_dict = {"data_matrix": np.array(sa_data_matrix), "structures": sa_strucs}

        # save with prefix if it is given
        if save_prefix is None:
            with open(f"sa_dict.pkl", "wb") as f:
                pickle.dump(sa_dict, f)

        else:
            with open(f"{save_prefix}_sa_{data_type}_dict.pkl", "wb") as f:
                pickle.dump(sa_dict, f)

    return np.array(sa_data_matrix), sa_strucs


def load_strain_matrix(strain_pkl):
    """Loads the strain data matrix and corresponding structures

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
    """

    # load the dictionary information
    db = pickle.load(open(f"{strain_pkl}", "rb"))
    sa_data_matrix = db["data_matrix"]
    sa_strucs = db["structures"]

    return np.array(sa_data_matrix), sa_strucs

def calculate_coverage_matching_scores(reference_strucs, sample_strucs, resnum_bounds, rmsd_threshold=1., simultaneous=True, verbose=False):
    '''Calculates the coverage and matching metrics for a sample set of structures/conformational ensemble compared to a reference set of structures/conformational ensemble.

    The coverage and matching metrics used are defined by Xu et al. (2021) ICLR. The coverage metric measures the diversity of the sample set compared to the reference set. The matching metric measures the similarity of the sample set to the reference set.

    Parameters:
    -----------
    reference_strucs : list of str
        Array containing the file paths to reference structures.

    sample_strucs : list of str
        Array containing the file paths to sample/generated structures.

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values.

    rmsd_threshold : float
        Minimum value for two structures to be considered similar.

    simultaneous : bool
        Indicator for simultaneous calculation of RMSD; otherwise, calculations are sequential.

    verbose : bool
        Indicator for verbose output.

    Returns:
    --------
    coverage : float
        Coverage score that compares the diversity of the supplied conformational ensembles.

    matching : float
        Matching score that compares the similarity of the supplied conformational ensembles.
    '''

    # create sorted atom list
    sorted_atom_list = [(res, at) for res in np.arange(resnum_bounds[0], resnum_bounds[1]+1) for at in ['N', 'CA', 'C']]

    # initialize coverage and matching score
    coverage = 0
    matching = 0

    # iterate through the reference structures
    ref_coords = list()
    for ref in reference_strucs:

        # load the reference structure coordinates
        ref_pstructure = PandasPdb().read_pdb(ref) if ref.endswith('.pdb') else PandasMmcif().read_mmcif(ref)
        ref_coords.append(coords_from_atoms(ref_pstructure.df['ATOM'], sorted_atom_list))

    if verbose:
        print("Loaded reference structure data!")

    # iterate through the sample structures
    sample_coords = list()
    for sample in sample_strucs:

        # load the sample structure coordinates
        sample_pstructure = PandasPdb().read_pdb(sample) if sample.endswith('.pdb') else PandasMmcif().read_mmcif(sample)
        sample_coords.append(coords_from_atoms(sample_pstructure.df['ATOM'], sorted_atom_list))

    if verbose:
        print("Loaded sample structure data!")

    # generate the comparisons
    comparisons = np.array(list(product(ref_coords, sample_coords))).reshape(len(ref_coords), len(sample_coords), 2, -1, 3)

    # calculate RMSDs
    if simultaneous:
        rmsds = np.sqrt(np.sum(np.sum(np.square(comparisons[:,:,0,:,:] - comparisons[:,:,1,:,:]), axis=2), axis=2) / len(sorted_atom_list))
    else:
        rmsds = np.zeros((len(ref_coords), len(sample_coords)))

        # iterate through the coordinate pairs
        for i in np.arange(len(ref_coords)):
            for j in np.arange(len(sample_coords)):
                rmsds[i,j] = np.sqrt(np.sum(np.square(comparisons[i,j,0,:,:] - comparisons[i,j,1,:,:])) / len(sorted_atom_list))

    # calculate coverage and matching scores
    coverage = np.sum(np.any(rmsds < rmsd_threshold, axis=1).astype('int')) / len(reference_strucs)
    matching = np.sum(np.min(rmsds, axis=1)) / len(reference_strucs)

    # print coverage and matching metrics
    if verbose:

        print(f'Coverage metric: {np.round(coverage*100, decimals=2)}%')
        print(f'Matching metric: {np.round(matching, decimals=3)}')

    return coverage, matching

def calculate_dh_pw(i, j, u_pca, pw_pca, resnum_bounds, psi_idx, phi_idx, omg_idx):
    return pearsonr(
        calculate_dh_rc(u_pca.components_[i, :])[psi_idx]
        + calculate_dh_rc(u_pca.components_[i, :])[phi_idx]
        + calculate_dh_rc(u_pca.components_[i, :])[omg_idx],
        calculate_pw_rc(pw_pca.components_[j, :], resnum_bounds)[1:],
    )[0]

def calculate_dh_sa(i, j, u_pca, sa_pca, shared_atom_set, psi_idx, phi_idx, omg_idx):
    return pearsonr(
        calculate_dh_rc(u_pca.components_[i, :])[psi_idx]
        + calculate_dh_rc(u_pca.components_[i, :])[phi_idx]
        + calculate_dh_rc(u_pca.components_[i, :])[omg_idx],
        calculate_sa_rc(sa_pca.components_[j, :], sorted(shared_atom_set))[1:],
    )[0]

def calculate_pw_sa(i, j, pw_pca, sa_pca, resnum_bounds, shared_atom_set):
    return pearsonr(
        calculate_pw_rc(pw_pca.components_[i, :], resnum_bounds),
        calculate_sa_rc(sa_pca.components_[j, :], sorted(shared_atom_set)),
    )[0]
