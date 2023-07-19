
from colav.extract_data import * 
from colav.internal_coordinates import * 
from colav.strain_analysis import * 
from biopandas.pdb import PandasPdb
from numpy.testing import assert_allclose
import pytest

def test_calculate_dihedral(): 

    # testing typical use cases
    assert calculate_dihedral([1,0,0], [0,0,0], [0,1,0], [1,1,0]) == 0
    assert calculate_dihedral([1,0,0], [0,0,0], [0,1,0], [-1,1,0]) == np.pi
    assert calculate_dihedral([1,0,0], [0,0,0], [0,1,0], [0,1,-1]) == np.pi/2

def test_calculate_internal_coordinates(): 

    # testing using TEST.pdb; see header of TEST.pdb for more information
    # because of conversions from degrees to radians and vice-versa, we have used rounding to appropriate decimal places
    # according to the standard_geometry.cif
    test_ppdb = PandasPdb().read_pdb('TEST.pdb')
    assert_allclose(np.round(np.rad2deg(calculate_backbone_dihedrals(test_ppdb, (1,5))),0), 4*[-47,-180,-57])
    assert_allclose(np.round(np.rad2deg(calculate_bond_angles(test_ppdb, (1,5))),1), [111.0]+4*[117.2,121.7,111.0])
    assert_allclose(np.round(calculate_bond_distances(test_ppdb, (1,5)),3), [1.459,1.525]+4*[1.336,1.459,1.525],rtol=1e-3)

def test_nerf_reconstruction(): 
    
    # testing using TEST.pdb see header of TEST.pdb for more information
    test_ppdb = PandasPdb().read_pdb('TEST.pdb')
    test_coords = test_ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
    dihedrals = calculate_backbone_dihedrals(test_ppdb, (1,5))
    angles = calculate_bond_angles(test_ppdb, (1,5))[1:]
    distances = calculate_bond_distances(test_ppdb, (1,5))[2:]
    assert_allclose(nerf_reconstruction(test_coords[:3,:], angles, distances, dihedrals), test_coords)

