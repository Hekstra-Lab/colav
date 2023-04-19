
import numpy as np 
from biopandas import PandasPdb
from scipy.spatial.distance import cdist


def calculate_dihedral(p0, p1, p2, p3): 
    '''
    Calculate the dihedral/torsion angle between the supplied four points using the praxeolytic formula
    '''

    # calculate vectors between points 
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    
    # normalize b1
    b1 /= np.linalg.norm(b1)
    
    # calculate vector projections
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    
    # calculate angle between v and w
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.arctan2(y, x)

def calculate_phiomgpsis(ppdb): 
    '''
    Calculate all (phi, omega, and psi) dihedral angles 
    The DataFrame should already be cropped by resnum_bounds
    '''

    # initialize storage
    phiomgpsis = list()

    # get only atoms that are relevant for the calculation 
    atoms = ["N", "CA", "C"]
    mainchain = ppdb.df['ATOM'][(ppdb.df['ATOM']['atom_name'] == 'N') |
                                (ppdb.df['ATOM']['atom_name'] == 'CA')|
                                (ppdb.df['ATOM']['atom_name'] == 'C')]

    # get indices of the "C" backbone atoms 
    idxs = df.loc[df.atom_id == "C"].index

    # iterate through the backbone atoms 
    for i,k in enumerate(idxs): 

        if i >= idxs.shape[0]-1: 
            continue

        # grab relevant coordinates for calculation 
        p0, p1, p2, p3, p4, p5 = df.iloc[k-2:k+4][["x", "y", "z"]].to_numpy()

        # calculate psi
        psi = calculate_dihedral(p0, p1, p2, p3)

        # calculate omega
        omega = calculate_dihedral(p1, p2, p3, p4)

        # calculate phi 
        phi = calculate_dihedral(p2, p3, p4, p5)

        phiomgpsis.extend([psi, omega, phi])

    return np.array(phiomgpsis)

def calculate_phipsis(df): 
    '''
    Calculate the phi and psi dihedral angles 
    '''
    
    # initialize storage array for phis and psis 
    phipsis = list()
    
    # determine resnum_bounds 
    resnum_bounds = (df.iloc[0].resnum, df.iloc[-1].resnum)
    
    # get indices of the "C" backbone atoms 
    idxs = df.loc[df.atom_id == "C"].index 
    
 
    for i,k in enumerate(idxs): 

        if i >= idxs.shape[0]-2: 
            continue
        
        # grab the relevant coordinates 
        p0, p1, p2, p3, p4 = df.iloc[k:k+5][["x", "y", "z"]].to_numpy()
                
        # calculate phi
        phi = calculate_dihedral(p0, p1, p2, p3)
        # calculate psi
        psi = calculate_dihedral(p1, p2, p3, p4)
        
        # store phi and psi values 
        phipsis.extend([phi, psi])
        
    return np.array(phipsis)

def calculate_omegas(df): 

    # initialize storage array for omegas 
    omgs = list()

    # determine resnum_bounds 
    resnum_bounds = (df.iloc[0].resnum, df.iloc[-1].resnum)

    # filter dataset 
    atoms = ["C", "N", "CA", "O"]
    df = get_data(df, atoms=atoms)

    # get indices of the "C" backbone atoms 
    idxs = df.loc[df.atom_id == "C"].index

    for i,k in enumerate(idxs): 

        print(df.iloc[k-2:k+5])
        # print(df.iloc[k-2:k+2][["x", "y", "z"]].to_numpy())

    return omgs

def calculate_bond_angles(df: pd.DataFrame, resnum_bounds): 
    '''
    assume that the supplied DataFrame only contains backbone (here N, CA, C) atoms 
    '''

    # get the desired coordinates for the current structure 
    coords,_ = get_coords(df=df, atoms=["N", "CA", "C"], resnum_bounds=resnum_bounds)

    # calculate and normalize the difference vectors between coordinates
    firsts = normalize(coords[1:-2] - coords[2:-1], axis=1)
    seconds = normalize(coords[3:] - coords[2:-1], axis=1)

    # initialize storage array for calculation 
    bond_angles = list()

    # calculate the angles between the difference vectors 
    for i in np.arange(firsts.shape[0]): 

        bond_angles.append(np.arccos(np.clip(np.dot(firsts[i,:], seconds[i,:]), -1.0, 1.0)))

    return np.array(bond_angles).astype("float64")

def calculate_bond_distances(df: pd.DataFrame, resnum_bounds): 
    '''
    assume that the supplied DataFrame only contains backbone (here N, CA, C) atoms 
    '''

    # get the desired coordinates for the current structure 
    coords,_ = get_coords(df=df, atoms=["N", "CA", "C"], resnum_bounds=resnum_bounds)

    # calculate the consecutive bond distances 
    bond_distances = np.diag(cdist(coords, coords), k=1)[2:]

    return np.array(bond_distances).astype("float64")

def reconstruction(initial_coords: np.array, bond_angles, bond_distances, dihedral_angles): 
    '''
    reconstruction implementation based on the NeRF algorithm for strictly backbone atoms (here N, CA, C)
    see reference in journal of computational chemistry Parsons et al 2005
    
    Users must supply the initial coordinates of the first amino acid to be used for the reconstruction; 
    this amino acid should be numbered as resnum_bounds[0]
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