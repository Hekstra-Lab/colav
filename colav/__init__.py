from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matchmaps")
except PackageNotFoundError:
    __version__ = "uninstalled"

from colav.extract_data import * 