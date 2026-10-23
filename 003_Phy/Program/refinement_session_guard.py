from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, messagebox

from curation_common import WORKING_ROOT, normalize_cluster_group, now_text, result_signature, validate_result, write_json
from refinement_common import REFINEMENT_DONE_MARKER, REFINEMENT_SESSION_MARKER, backup_refinement

def dialog(kind: str, title: str, message: str):
    root=Tk(); root.withdraw(); root.attributes("-topmost",True)
    try:
        if kind=="yesno": return messagebox.askyesno(title,message,parent=root)
        getattr(messagebox,{"error":"showerror","warning":"showwarning"}.get(kind,"showinfo"))(title,message,parent=root)
    finally: root.destroy()

def main() -> int:
    parser=argparse.ArgumentParser(description="Guard a final Phy refinement session with all clusters visible.")
    parser.add_argument("--result-folder",required=True,type=Path); args=parser.parse_args()
    folder=args.result_folder.resolve(); params=folder/"params.py"
    try: folder.relative_to(WORKING_ROOT.resolve())
    except ValueError:
        dialog("error","Final refinement could not start","Final refinement may modify only results under Working_Curated.")
        return 2
    if not params.is_file():
        dialog("error","Final refinement could not start",f"Missing params.py:\n{folder}"); return 2
    try:
        normalize_cluster_group(folder,backup=True)
        before_validation=validate_result(folder); before=result_signature(folder); backup=backup_refinement(folder)
    except Exception as error:
        dialog("error","Final-refinement preflight failed",str(error)); return 3
    session={"stage":"phy_open","started":now_text(),"backup":str(backup),"before":before,"validation_before":before_validation,
             "mode":"All clusters by channel"}
    write_json(folder/REFINEMENT_SESSION_MARKER,session)
    command=[sys.executable,str(Path(__file__).with_name("refinement_phy.py")),str(params)]
    exit_code=subprocess.call(command,cwd=folder)
    try:
        after=result_signature(folder); validation=validate_result(folder)
    except Exception as error:
        write_json(folder/REFINEMENT_SESSION_MARKER,{**session,"stage":"save_validation_failed","finished":now_text(),"error":str(error)})
        dialog("error","Final-result validation failed",f"The saved result failed integrity validation.\n\nBackup: {backup}\n\n{error}"); return 4
    changed=[name for name in before if before[name]!=after[name]]
    write_json(folder/REFINEMENT_SESSION_MARKER,{**session,"stage":"phy_closed","finished":now_text(),"exit_code":exit_code,
               "changed_authoritative_files":changed,"after":after,"validation_after":validation})
    confirmed=bool(changed)
    if not changed:
        confirmed=bool(dialog("yesno","No file changes detected","No authoritative result file changed.\n\nIf all clusters were reviewed and no edits were required, mark this recording as complete?"))
    if confirmed:
        marker={"stage":"final_refinement_completed","completed":now_text(),"changed_authoritative_files":changed,
                "backup":str(backup),"signature":after,"validation":validation,
                "visible_filter":"All clusters; channel ascending"}
        write_json(folder/REFINEMENT_DONE_MARKER,marker)
        dialog("info","Final refinement completed",f"The result was saved and marked complete.\n\nClusters: {validation['cluster_count']}, Good: {validation['good_count']}, Noise: {validation['noise_count']}")
        return 0
    return exit_code if exit_code else 5

if __name__=="__main__": raise SystemExit(main())
