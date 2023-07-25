from colav.extract_data import *
from colav.internal_coordinates import *
from colav.strain_analysis import *
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import pdist
from numpy.testing import assert_allclose, assert_array_equal
import pytest


def test_calculate_dihedral():
    # testing typical use cases
    assert calculate_dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]) == 0
    assert calculate_dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [-1, 1, 0]) == np.pi
    assert calculate_dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, -1]) == np.pi / 2


def test_calculate_internal_coordinates():
    # testing using TEST1.pdb; see header of TEST1.pdb for more information
    # because of conversions from degrees to radians and vice-versa, we have used rounding to appropriate decimal places
    # according to the standard_geometry.cif
    test_ppdb = PandasPdb().read_pdb("tests/TEST1.pdb")
    assert_allclose(
        np.round(np.rad2deg(calculate_backbone_dihedrals(test_ppdb, (1, 5))), 0),
        4 * [-47, -180, -57],
    )
    assert_allclose(
        np.round(np.rad2deg(calculate_bond_angles(test_ppdb, (1, 5))), 1),
        [111.0] + 4 * [117.2, 121.7, 111.0],
    )
    assert_allclose(
        np.round(calculate_bond_distances(test_ppdb, (1, 5)), 3),
        [1.459, 1.525] + 4 * [1.336, 1.459, 1.525],
        rtol=1e-3,
    )


def test_nerf_reconstruction():
    # testing using TEST1.pdb; see header of TEST1.pdb for more information
    test_ppdb = PandasPdb().read_pdb("tests/TEST1.pdb")
    test_coords = test_ppdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy()
    dihedrals = calculate_backbone_dihedrals(test_ppdb, (1, 5))
    angles = calculate_bond_angles(test_ppdb, (1, 5))[1:]
    distances = calculate_bond_distances(test_ppdb, (1, 5))[2:]
    assert_allclose(
        nerf_reconstruction(test_coords[:3, :], angles, distances, dihedrals),
        test_coords,
    )


def test_determine_shared_atoms():
    # testing using TEST1.pdb and TEST2.pdb; see header of TEST1.pdb for more information
    test_ppdbdf = PandasPdb().read_pdb("tests/TEST1.pdb").df["ATOM"]
    assert determine_shared_atoms(
        ["tests/TEST1.pdb", "tests/TEST2.pdb"],
        test_ppdbdf,
        (1, 5),
        atoms=["N", "CA", "C"],
        alt_locs=[""],
    ) == (
        set(
            [
                (1, "N"),
                (1, "CA"),
                (1, "C"),
                (2, "N"),
                (2, "CA"),
                (2, "C"),
                (3, "N"),
                (3, "CA"),
                (3, "C"),
                (4, "N"),
                (4, "CA"),
                (4, "C"),
                (5, "N"),
                (5, "CA"),
                (5, "C"),
            ]
        ),
        ["tests/TEST1.pdb", "tests/TEST2.pdb"],
    )


def test_data_from_atoms():
    # at some point in the future, I would like to write a more general function for this purpose
    # testing using TEST1.pdb; see header of TEST1.pdb for more information
    test_ppdbdf = PandasPdb().read_pdb("tests/TEST1.pdb").df["ATOM"]
    assert_array_equal(
        coords_from_atoms(
            test_ppdbdf,
            [
                (1, "N"),
                (1, "CA"),
                (1, "C"),
                (2, "N"),
                (2, "CA"),
                (2, "C"),
                (3, "N"),
                (3, "CA"),
                (3, "C"),
                (4, "N"),
                (4, "CA"),
                (4, "C"),
                (5, "N"),
                (5, "CA"),
                (5, "C"),
            ],
        ),
        np.array(
            [
                [0.000, 0.000, 0.000],
                [1.459, 0.000, 0.000],
                [2.006, 0.000, 1.424],
                [1.468, 0.869, 2.285],
                [1.917, 0.953, 3.670],
                [1.784, -0.396, 4.370],
                [0.630, -1.052, 4.217],
                [0.397, -2.349, 4.843],
                [1.459, -3.358, 4.420],
                [1.746, -3.434, 3.117],
                [2.745, -4.365, 2.605],
                [4.097, -4.136, 3.271],
                [4.531, -2.876, 3.356],
                [5.811, -2.545, 3.974],
                [5.874, -3.059, 5.409],
            ]
        ),
    )
    assert_array_equal(
        bfacs_from_atoms(
            test_ppdbdf,
            [
                (1, "N"),
                (1, "CA"),
                (1, "C"),
                (2, "N"),
                (2, "CA"),
                (2, "C"),
                (3, "N"),
                (3, "CA"),
                (3, "C"),
                (4, "N"),
                (4, "CA"),
                (4, "C"),
                (5, "N"),
                (5, "CA"),
                (5, "C"),
            ],
        ),
        np.array(15 * [40.00]),
    )


def test_calculate_strain():
    # testing using TEST1.pdb and TEST2.pdb; see header of TEST1.pdb for more information
    ref_coords = (
        PandasPdb()
        .read_pdb("tests/TEST3.pdb")
        .df["ATOM"][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
    )
    def_coords = (
        PandasPdb()
        .read_pdb("tests/TEST4.pdb")
        .df["ATOM"][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
    )
    shear_energy, shear_tensor, strain_tensor = calculate_strain(ref_coords, def_coords)
    assert_allclose(
        shear_energy, np.array([1.34821937, 1.34821937, 1.34821937, 1.34821937])
    )
    assert_allclose(
        shear_tensor,
        np.array(
            [
                [
                    [2.63130473e-01, -6.37512148e-01, -1.10977327e-15],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [-1.15757414e-15, 2.44897959e-01, 2.63130473e-01],
                ],
                [
                    [2.63130473e-01, -6.37512148e-01, 1.03164408e-17],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [3.54887521e-17, 2.44897959e-01, 2.63130473e-01],
                ],
                [
                    [2.63130473e-01, -6.37512148e-01, 5.87481120e-17],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [1.41403949e-16, 2.44897959e-01, 2.63130473e-01],
                ],
                [
                    [2.63130473e-01, -6.37512148e-01, -1.35179128e-15],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [-1.27616516e-15, 2.44897959e-01, 2.63130473e-01],
                ],
            ]
        ),
        atol=1e-7,
    )
    assert_allclose(
        strain_tensor,
        np.array(
            [
                [
                    [3.88578059e-16, -6.37512148e-01, -1.10977327e-15],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [-1.15757414e-15, 2.44897959e-01, 1.11022302e-16],
                ],
                [
                    [0.00000000e00, -6.37512148e-01, 1.03164408e-17],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [3.54887521e-17, 2.44897959e-01, 0.00000000e00],
                ],
                [
                    [6.66133815e-16, -6.37512148e-01, 5.87481120e-17],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [1.41403949e-16, 2.44897959e-01, 5.55111512e-16],
                ],
                [
                    [7.77156117e-16, -6.37512148e-01, -1.35179128e-15],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [-1.27616516e-15, 2.44897959e-01, 1.33226763e-15],
                ],
            ]
        ),
        atol=1e-7,
    )


def test_calculate_strain_dict():
    # testing using TEST3.pdb and TEST4.pdb; see header of TEST1.pdb for more information
    strain_dict, shared_atom_set = calculate_strain_dict(
        ["tests/TEST3.pdb", "tests/TEST4.pdb"],
        "tests/TEST3.pdb",
        (1, 2),
        atoms=["N", "CA", "C"],
        alt_locs=[""],
        save=False,
    )
    assert shared_atom_set == set([(1, "N"), (1, "CA"), (1, "C"), (2, "N")])
    assert_allclose(
        strain_dict["tests/TEST4.pdb"]["sheare"],
        np.array([1.34821937, 1.34821937, 1.34821937, 1.34821937]),
    )
    assert_allclose(
        strain_dict["tests/TEST4.pdb"]["sheart"],
        np.array(
            [
                [
                    [2.63130473e-01, -6.37512148e-01, -1.10977327e-15],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [-1.15757414e-15, 2.44897959e-01, 2.63130473e-01],
                ],
                [
                    [2.63130473e-01, -6.37512148e-01, 1.03164408e-17],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [3.54887521e-17, 2.44897959e-01, 2.63130473e-01],
                ],
                [
                    [2.63130473e-01, -6.37512148e-01, 5.87481120e-17],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [1.41403949e-16, 2.44897959e-01, 2.63130473e-01],
                ],
                [
                    [2.63130473e-01, -6.37512148e-01, -1.35179128e-15],
                    [-6.37512148e-01, -5.26260946e-01, 2.44897959e-01],
                    [-1.27616516e-15, 2.44897959e-01, 2.63130473e-01],
                ],
            ]
        ),
        atol=1e-10,
    )
    assert_allclose(
        strain_dict["tests/TEST4.pdb"]["straint"],
        np.array(
            [
                [
                    [3.88578059e-16, -6.37512148e-01, -1.10977327e-15],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [-1.15757414e-15, 2.44897959e-01, 1.11022302e-16],
                ],
                [
                    [0.00000000e00, -6.37512148e-01, 1.03164408e-17],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [3.54887521e-17, 2.44897959e-01, 0.00000000e00],
                ],
                [
                    [6.66133815e-16, -6.37512148e-01, 5.87481120e-17],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [1.41403949e-16, 2.44897959e-01, 5.55111512e-16],
                ],
                [
                    [7.77156117e-16, -6.37512148e-01, -1.35179128e-15],
                    [-6.37512148e-01, -7.89391419e-01, 2.44897959e-01],
                    [-1.27616516e-15, 2.44897959e-01, 1.33226763e-15],
                ],
            ]
        ),
        atol=1e-10,
    )
    assert_array_equal(strain_dict["tests/TEST4.pdb"]["bfacs"], 4 * [40.00])
    assert strain_dict["tests/TEST4.pdb"]["atom_list"] == sorted(
        [(1, "N"), (1, "CA"), (1, "C"), (2, "N")]
    )
    assert strain_dict["tests/TEST4.pdb"]["atom_idxs"] == [0, 1, 2, 3]


def test_calculate_tl():
    # calculate mock transformed loadings using arbitrary raw loadings
    assert_array_equal(
        calculate_dh_tl([1.0, 0.5, -1.0, -0.5, -1.2, -0.8]), [1.5, 1.7, 1.8]
    )
    assert_array_equal(
        calculate_pw_tl([1.0, 0.5, -1.0, -0.5, -1.2, -0.8], (1, 4)), [2.5, 2.7, 1.8, 3]
    )
    assert_array_equal(
        calculate_sa_tl(
            [1.0, 0.5, -1.0, -0.5, -1.2, -0.8, 1.0, 0.5, -1.0, -0.5, -1.2, -0.8],
            [(1, "N"), (1, "CA"), (1, "C"), (2, "N")],
        ),
        [7.5, 2.5],
    )


def test_load_matrix():
    resnum_bounds = (1, 5)
    # load matrices and check that the loaded dictionaries have the correct structure
    ############### DIHEDRAL ANGLES ###############
    dh_data_matrix, dh_strucs = load_dihedral_matrix("tests/test_dh_dict.pkl")
    assert_allclose(
        np.abs(np.round(np.rad2deg(dh_data_matrix), 0)),
        np.abs(np.vstack([4 * [-47.0, -180.0, -57.0], 4 * [120.0, 180.0, -120.0]])),
    )
    assert dh_strucs == ["tests/TEST1.pdb", "tests/TEST2.pdb"]
    ############### PAIRWISE DISTANCES ###############
    pw_data_matrix, pw_strucs = load_pw_matrix("tests/test_pw_dict.pkl")
    test1_ppdb = PandasPdb().read_pdb("tests/TEST1.pdb")
    test1_cas = test1_ppdb.df["ATOM"][
        (test1_ppdb.df["ATOM"]["atom_name"] == "CA")
        & (test1_ppdb.df["ATOM"]["residue_number"] >= resnum_bounds[0])
        & (test1_ppdb.df["ATOM"]["residue_number"] <= resnum_bounds[1])
        & (test1_ppdb.df["ATOM"]["alt_loc"] == "")
    ]
    test2_ppdb = PandasPdb().read_pdb("tests/TEST2.pdb")
    test2_cas = test2_ppdb.df["ATOM"][
        (test2_ppdb.df["ATOM"]["atom_name"] == "CA")
        & (test2_ppdb.df["ATOM"]["residue_number"] >= resnum_bounds[0])
        & (test2_ppdb.df["ATOM"]["residue_number"] <= resnum_bounds[1])
        & (test2_ppdb.df["ATOM"]["alt_loc"] == "")
    ]
    assert_allclose(
        pw_data_matrix,
        np.vstack(
            [
                pdist(test1_cas[["x_coord", "y_coord", "z_coord"]].to_numpy()),
                pdist(test2_cas[["x_coord", "y_coord", "z_coord"]].to_numpy()),
            ]
        ),
    )
    assert pw_strucs == ["tests/TEST1.pdb", "tests/TEST2.pdb"]
    ############### STRAIN ANALYSIS ###############
    sa_data_matrix, sa_strucs = load_strain_matrix("tests/test_sa_sheart_dict.pkl")
    correction = np.sqrt(np.array(12 * [40.00]))
    test3_coords = (
        PandasPdb()
        .read_pdb("tests/TEST3.pdb")
        .df["ATOM"][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
    )
    test4_coords = (
        PandasPdb()
        .read_pdb("tests/TEST4.pdb")
        .df["ATOM"][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
    )
    shear_energy, shear_tensor, strain_tensor = calculate_strain(
        test3_coords, test4_coords
    )
    shear_data = np.array(
        [tensor[np.triu_indices(3, 1)] for tensor in shear_tensor]
    ).flatten()
    processed = shear_data / correction
    assert_allclose(sa_data_matrix, processed[None, :], atol=1e-9)
    assert sa_strucs == ["tests/TEST4.pdb"]


def test_generate_matrix():
    resnum_bounds = (1, 5)
    # testing using TEST1.pdb and TEST2.pdb; see header of TEST1.pdb for more information
    ############### DIHEDRAL ANGLES ###############
    dh_data_matrix, dh_strucs = generate_dihedral_matrix(
        ["tests/TEST1.pdb", "tests/TEST2.pdb"],
        resnum_bounds,
        save=False,
        save_prefix="tests/test",
    )
    # used an abs because the only sign flip occurs with 180 or -180
    assert_allclose(
        np.abs(np.round(np.rad2deg(dh_data_matrix), 0)),
        np.abs(np.vstack([4 * [-47.0, -180.0, -57.0], 4 * [120.0, 180.0, -120.0]])),
    )
    assert dh_strucs == ["tests/TEST1.pdb", "tests/TEST2.pdb"]
    ############### PAIRWISE DISTANCES ###############
    pw_data_matrix, pw_strucs = generate_pw_matrix(
        ["tests/TEST1.pdb", "tests/TEST2.pdb"],
        resnum_bounds,
        save=False,
        save_prefix="tests/test",
    )
    test1_ppdb = PandasPdb().read_pdb("tests/TEST1.pdb")
    test1_cas = test1_ppdb.df["ATOM"][
        (test1_ppdb.df["ATOM"]["atom_name"] == "CA")
        & (test1_ppdb.df["ATOM"]["residue_number"] >= resnum_bounds[0])
        & (test1_ppdb.df["ATOM"]["residue_number"] <= resnum_bounds[1])
        & (test1_ppdb.df["ATOM"]["alt_loc"] == "")
    ]
    test2_ppdb = PandasPdb().read_pdb("tests/TEST2.pdb")
    test2_cas = test2_ppdb.df["ATOM"][
        (test2_ppdb.df["ATOM"]["atom_name"] == "CA")
        & (test2_ppdb.df["ATOM"]["residue_number"] >= resnum_bounds[0])
        & (test2_ppdb.df["ATOM"]["residue_number"] <= resnum_bounds[1])
        & (test2_ppdb.df["ATOM"]["alt_loc"] == "")
    ]
    assert_allclose(
        pw_data_matrix,
        np.vstack(
            [
                pdist(test1_cas[["x_coord", "y_coord", "z_coord"]].to_numpy()),
                pdist(test2_cas[["x_coord", "y_coord", "z_coord"]].to_numpy()),
            ]
        ),
    )
    assert pw_strucs == ["tests/TEST1.pdb", "tests/TEST2.pdb"]
    # testing using TEST3.pdb and TEST4.pdb; see header of TEST1.pdb for more information
    ############### STRAIN ANALYSIS ###############
    sa_data_matrix, sa_strucs = generate_strain_matrix(
        ["tests/TEST3.pdb", "tests/TEST4.pdb"],
        "tests/TEST3.pdb",
        "sheart",
        (1, 2),
        atoms=["N", "CA", "C"],
        alt_locs=[""],
        save=False,
        save_prefix="tests/test",
    )
    correction = np.sqrt(np.array(12 * [40.00]))
    test3_coords = (
        PandasPdb()
        .read_pdb("tests/TEST3.pdb")
        .df["ATOM"][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
    )
    test4_coords = (
        PandasPdb()
        .read_pdb("tests/TEST4.pdb")
        .df["ATOM"][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
    )
    shear_energy, shear_tensor, strain_tensor = calculate_strain(
        test3_coords, test4_coords
    )
    shear_data = np.array(
        [tensor[np.triu_indices(3, 1)] for tensor in shear_tensor]
    ).flatten()
    processed = shear_data / correction
    assert_allclose(sa_data_matrix, processed[None, :], atol=1e-9)
    assert sa_strucs == ["tests/TEST4.pdb"]
