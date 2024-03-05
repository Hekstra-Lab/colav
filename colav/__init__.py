from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("colav")
except PackageNotFoundError:
    __version__ = "uninstalled"

from colav.extract_data import (
    calculate_dh_rc, 
    calculate_pw_rc, 
    calculate_sa_rc, 
    generate_dihedral_matrix, 
    generate_pw_matrix, 
    generate_strain_matrix, 
    load_dihedral_matrix, 
    load_pw_matrix, 
    load_strain_matrix, 
    calculate_coverage_matching_scores, 
    )