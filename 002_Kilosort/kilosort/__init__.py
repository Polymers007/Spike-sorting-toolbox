
import importlib.metadata
try:
    __version__ = importlib.metadata.version('kilosort')
except importlib.metadata.PackageNotFoundError:
    
    pass

from .utils import PROBE_DIR, DOWNLOADS_DIR
from .run_kilosort import run_kilosort
from .parameters import DEFAULT_SETTINGS
