
import numpy as np 
from scipy.spatial.distance import cdist

def calculate_dihedral(p0, p1, p2, p3) -> float: 
    '''Calculates the dihedral angle between four points in three dimensions. 

    Returns the dihedral angle. 

    Parameters: 
    -----------
    p0 : array_like
        First coordinate defining the plane. 
    
    p1 : array_like
        Second coordinate defining the plane.
    
    p2 : array_like
        Third coordinate defining the plane.
    
    p3 : array_like
        Fourth coordinate defining the plane.
    
    Returns: 
    --------
    dihedral_angle : float
        Computed dihedral angle (radians) between given points. 

    '''

    # convert the input points to np arrays 
    p0, p1, p2, p3 = np.array(p0), np.array(p1), np.array(p2), np.array(p3)

    # calculate vectors between points 
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalize b1
    b1 = b1 / np.linalg.norm(b1)
    
    # calculate vector projections
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    
    # calculate angle between v and w
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.arctan2(y, x)

def calculate_backbone_dihedrals(ppdb, resnum_bounds, no_psi=False, no_omega=False, no_phi=False, verbose=False) -> np.array: 
    '''Calculates backbone dihedral angles psi, omega, and phi (returned in that
    order) of consecutive residues. 

    Returns the desired dihedral angles in order between `resnum_bounds`
    as an array. 

    Parameters: 
    -----------
    ppdb : PandasPdb
        Dataframe containing the target protein structure (preferably one chain). 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values 
        for calculating the desired dihedral angles. 

    no_psi : bool, optional 
        Indicator to exclude psi dihedral angle from returned dihedral angles.

    no_omega : bool, optional 
        Indicator to exclude omega dihedral angle from returned dihedral angles. 

    no_phi : bool, optional 
        Indicator to exclude phi dihedral angle from returne dihedral angles. 

    verbose : bool, optional 
        Indicator for verbose output. 

    Returns: 
    --------
    dihedrals : array_like
        Desired dihedral angles in order between `resnum_bounds`. 

    '''

    # initialize storage
    dihedrals = list()

    # get only atoms that are relevant for the calculation 
    mainchain = ppdb.df['ATOM'].loc[(ppdb.df['ATOM']['atom_name'] == 'N') | # choose the correct atoms 
                                    (ppdb.df['ATOM']['atom_name'] == 'CA')|
                                    (ppdb.df['ATOM']['atom_name'] == 'C')]
    mainchain = mainchain.loc[(mainchain['residue_number'] >= resnum_bounds[0]) & # choose the correct residue numbers 
                              (mainchain['residue_number'] <= resnum_bounds[1])]
    mainchain = mainchain.loc[(mainchain['alt_loc'] == '') |  # choose the A alt_loc if there are any 
                              (mainchain['alt_loc'] == 'A')]
    mainchain = mainchain.reset_index()

    # get indices of the "C" backbone atoms 
    idxs = mainchain.loc[mainchain['atom_name'] == "C"].index

    # iterate through the backbone atoms 
    for i,k in enumerate(idxs): 

        if i >= idxs.shape[0]-1: 
            continue

        # grab relevant coordinates for calculation 
        p0, p1, p2, p3, p4, p5 = mainchain.iloc[k-2:k+4][['x_coord', 'y_coord', 'z_coord']].to_numpy()

        # calculate psi
        psi = calculate_dihedral(p0, p1, p2, p3)

        # calculate omega
        omega = calculate_dihedral(p1, p2, p3, p4)

        # calculate phi 
        phi = calculate_dihedral(p2, p3, p4, p5)

        # extend with desired dihedral angles only 
        dihedral_tmp = list()
        if not no_psi: 
            dihedral_tmp.append(psi)
        if not no_omega: 
            dihedral_tmp.append(omega)
        if not no_phi: 
            dihedral_tmp.append(phi)
        dihedrals.extend(dihedral_tmp)

    return np.array(dihedrals).astype('float64')

def calculate_bond_angles(ppdb, resnum_bounds) -> np.array: 
    '''Calculates the bond angles between consecutive residues. 

    Returns the bond angles in order between `resnum_bounds`. 
    
    Parameters: 
    -----------
    ppdb : PandasPdb
        Dataframe containing the target protein structure (preferably one chain). 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values 
        for calculating the desired dihedral angles

    Returns: 
    --------
    bond_angles : array_like
        Desired bond angles in order between `resnum_bounds`. 

    '''

    # get the desired coordinates for the current structure 
    mainchain = ppdb.df['ATOM'].loc[(ppdb.df['ATOM']['atom_name'] == 'N') | # choose the correct atoms 
                                    (ppdb.df['ATOM']['atom_name'] == 'CA')|
                                    (ppdb.df['ATOM']['atom_name'] == 'C')
                                   ]
    mainchain = mainchain.loc[(mainchain['residue_number'] >= resnum_bounds[0]) & # choose the correct residue numbers 
                              (mainchain['residue_number'] <= resnum_bounds[1])
                             ]
    mainchain = mainchain.loc[(mainchain['alt_loc'] == '') |  # choose the A alt_loc if there are any 
                              (mainchain['alt_loc'] == 'A')]
    mainchain = mainchain.reset_index()
    mainchain_coords = mainchain[['x_coord', 'y_coord', 'z_coord']].to_numpy()

    # calculate and normalize the difference vectors between coordinates
    firsts = (mainchain_coords[:-2] - mainchain_coords[1:-1]) / \
        np.hstack([
            np.linalg.norm(mainchain_coords[:-2] - mainchain_coords[1:-1], axis=1)[:,None],
            np.linalg.norm(mainchain_coords[:-2] - mainchain_coords[1:-1], axis=1)[:,None],
            np.linalg.norm(mainchain_coords[:-2] - mainchain_coords[1:-1], axis=1)[:,None]
        ])
    seconds = (mainchain_coords[2:] - mainchain_coords[1:-1]) / \
        np.hstack([
            np.linalg.norm(mainchain_coords[2:] - mainchain_coords[1:-1], axis=1)[:,None],
            np.linalg.norm(mainchain_coords[2:] - mainchain_coords[1:-1], axis=1)[:,None],
            np.linalg.norm(mainchain_coords[2:] - mainchain_coords[1:-1], axis=1)[:,None]
        ])

    # initialize storage array for calculation 
    bond_angles = list()

    # calculate the angles between the difference vectors 
    for i in np.arange(firsts.shape[0]): 

        bond_angles.append(np.arccos(np.clip(np.dot(firsts[i,:], seconds[i,:]), -1.0, 1.0)))

    return np.array(bond_angles).astype("float64")

def calculate_bond_distances(ppdb, resnum_bounds) -> np.array: 
    '''Calculate the bond distances between consecutive residues. 

    Returns the bond distances in order between `resnum_bounds`. 
    
    Parameters: 
    -----------
    ppdb : BioPandas DataFrame
        Dataframe containing the target protein structure (preferably one chain). 

    resnum_bounds : tuple
        Tuple containing the minimum and maximum (inclusive) residue number values 
        for calculating the desired bond distances. 

    Returns: 
    --------
    bond_distances : array_like
        Desired bond distances in order between `resnum_bounds`. 

    '''

    # get the desired coordinates for the current structure 
    mainchain = ppdb.df['ATOM'].loc[(ppdb.df['ATOM']['atom_name'] == 'N') | # choose the correct atoms 
                                    (ppdb.df['ATOM']['atom_name'] == 'CA')|
                                    (ppdb.df['ATOM']['atom_name'] == 'C')
                                   ]
    mainchain = mainchain.loc[(mainchain['residue_number'] >= resnum_bounds[0]) & # choose the correct residue numbers 
                              (mainchain['residue_number'] <= resnum_bounds[1])
                             ]
    mainchain = mainchain.loc[(mainchain['alt_loc'] == '') |  # choose the A alt_loc if there are any 
                              (mainchain['alt_loc'] == 'A')]
    mainchain = mainchain.reset_index()
    mainchain_coords = mainchain[['x_coord', 'y_coord', 'z_coord']].to_numpy()

    # calculate the consecutive bond distances 
    bond_distances = np.diag(cdist(mainchain_coords, mainchain_coords), k=1)

    return np.array(bond_distances).astype("float64")

def nerf_reconstruction(initial_coords, bond_angles, bond_distances, dihedral_angles) -> np.array: 
    '''Computes Cartesian coordinates of a protein using internal coordinates. 
    
    Computes the Cartesian coordinates of a polymer using an internal coordinate set 
    of bond angles, bond distances, and dihedral angles. The algorithm is an 
    implementation of the natural-extension reference frame (NeRF) algorithm for protein backbone 
    atom reconstruction using N, CA, and C. 
    
    See Parsons et al. (2005) in the Journal of Computational Chemistry for more information. 

    Returns Cartesian coordinates (N x 3) for the protein backbone. 

    Parameters:
    -----------
    initial_coords : array_like
        Initial coordinates of the first three atoms to initiate backbone reconstruction. 

    bond_angles : array_like
        Bond angles between consecutive residues. 

    bond_distances : array_like
        Bond distances bewteen consecutive residues. 

    dihedral_angles : array_like
        Dihedral angles between consecutive residues. 

    Usage: 
    ------
    All internal coordinate arguments (e.g., `bond_angles`, `bond_distances`, and `dihedral_angles`) 
    are required, and the size of the internal coordinate arrays must match AKA they must all 
    be of size N. 
    
    Returns: 
    --------
    mainchain_coords : array_like
        Cartesian coordinates determined by the supplied internal coordinates. 

    '''

    # initialize a list for collecting coordinates and dealing with coordinates 
    coord_list = list(initial_coords.flatten())
    current_coords = initial_coords # will be updated after each pass 

    # make sure that the reconstruction parameters are all of the same length 
    assert(bond_angles.shape == bond_distances.shape)
    assert(bond_angles.shape == dihedral_angles.shape)
    assert(bond_distances.shape == dihedral_angles.shape)

    bond_angles = np.pi - bond_angles

    # iterate throught the supplied data to reconstruct the backbone 
    for i in np.arange(bond_angles.shape[0]): 

        # calculate the D_2 vector 
        D2 = bond_distances[i] * np.array([np.cos(bond_angles[i]), np.cos(dihedral_angles[i]) * np.sin(bond_angles[i]), np.sin(dihedral_angles[i]) * np.sin(bond_angles[i])])

        # calculate useful intermediate values 
        bc = current_coords[2,:] - current_coords[1,:]
        bc_hat = bc / np.linalg.norm(bc)
        ab = current_coords[1,:] - current_coords[0,:]
        n = np.cross(ab, bc_hat)
        n_hat = n / np.linalg.norm(n)
        tmp = np.cross(n_hat, bc_hat) / np.linalg.norm(np.cross(n_hat, bc_hat))

        # calculate the transformation matrix M
        M = np.hstack([bc_hat[:,None], tmp[:,None], n_hat[:,None]]).T

        # calculate the coordinate D and extend coord_list
        D = D2 @ M + current_coords[2,:]
        coord_list.extend(D)

        # update the current_coords
        current_coords = np.vstack([current_coords[1:,:], D])

    return np.array(coord_list).astype("float64").reshape(-1, 3)
