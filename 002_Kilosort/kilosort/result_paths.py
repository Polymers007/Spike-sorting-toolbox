from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BINARY_ROOT = PROJECT_ROOT / "001_Bin" / "Data"
RESULT_ROOT = PROJECT_ROOT / "004_Result" / "Kilosort4"

def default_result_directory(filename=None):
    if filename is None:
        return RESULT_ROOT
    if isinstance(filename, (list, tuple)):
        if not filename:
            return RESULT_ROOT
        filename = filename[0]
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(BINARY_ROOT.resolve())
        device = relative.parent if relative.parent != Path(".") else Path("Default")
    except ValueError:
        device = Path(path.parent.name or "Default")
    return RESULT_ROOT / device / path.stem
