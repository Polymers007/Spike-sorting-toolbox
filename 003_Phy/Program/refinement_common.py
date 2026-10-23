from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from curation_common import AUTHORITATIVE_FILES, OUTPUT_ROOT, now_text, result_signature, write_json

REFINEMENT_SESSION_MARKER = "FINAL_REFINEMENT_SESSION.json"
REFINEMENT_DONE_MARKER = "FINAL_REFINEMENT_DONE.json"
REFINEMENT_BACKUP_ROOT = OUTPUT_ROOT / "Snapshots" / "Final_Refinement"
REFINEMENT_CANDIDATE_FILE = "cluster_refinement_candidate.tsv"
NOISE_RESCUE_MARKER = "NOISE_RESCUE_DONE.json"

def refinement_status(folder: Path) -> tuple[str, dict]:
    session_path = folder / REFINEMENT_SESSION_MARKER
    done_path = folder / REFINEMENT_DONE_MARKER
    from curation_common import read_json
    session = read_json(session_path)
    if session.get("stage") == "phy_open":
        return "In progress", session
    done = read_json(done_path)
    if done:
        if done.get("signature") == result_signature(folder):
            return "Completed", done
        return "Results changed", done
    return "Not refined", {}

def backup_refinement(folder: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = REFINEMENT_BACKUP_ROOT / folder.parent.name / folder.name / stamp
    destination.mkdir(parents=True, exist_ok=False)
    names = AUTHORITATIVE_FILES + (
        REFINEMENT_CANDIDATE_FILE,
        NOISE_RESCUE_MARKER,
        REFINEMENT_SESSION_MARKER,
        REFINEMENT_DONE_MARKER,
    )
    for name in names:
        source = folder / name
        if source.is_file(): shutil.copy2(source, destination / name)
    write_json(destination / "signature.json", {
        "created": now_text(), "working_folder": str(folder), "signature": result_signature(folder),
    })
    return destination
