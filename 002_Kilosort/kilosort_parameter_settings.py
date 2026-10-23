from __future__ import annotations

import ast
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from qtpy import QtCore

from kilosort.parameters import DEFAULT_SETTINGS

PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROGRAM_DIR.parent
PARAMETER_FILE = PROGRAM_DIR / "kilosort_batch_configuration.json"
BATCH_OPTION_DEFAULTS = {
    "do_CAR": True,
    "invert_sign": False,
    "clear_cache": False,
    "save_preprocessed_copy": False,
    "verbose_log": False,
}

def parse_value(text):
    value = text.strip()
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def coerce(value, default):
    if default is None:
        return value
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered not in {"true", "false"}:
                raise ValueError("expected True or False")
            return lowered == "true"
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value

def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value

def display_value(value):
    if isinstance(value, str):
        return value
    return repr(json_value(value))

def load_last_values():
    cache = QtCore.QSettings("Janelia", "Kilosort4")
    settings = {}
    for key, default in DEFAULT_SETTINGS.items():
        raw = cache.value(key) if cache.contains(key) else default
        settings[key] = parse_value(raw) if isinstance(raw, str) else raw
    values = {
        "settings": settings,
        "data_dtype": cache.value("data_dtype", "int16"),
        "waveform_scale_uV_per_bit": 1.0,
        "bad_channels": cache.value("bad_channels", []),
        "shank_idx": cache.value("shank_idx", None),
        "batch_options": BATCH_OPTION_DEFAULTS.copy(),
    }
    if PARAMETER_FILE.is_file():
        saved = json.loads(PARAMETER_FILE.read_text(encoding="utf-8"))
        values["settings"].update(saved.get("settings", {}))
        for key in ("data_dtype", "bad_channels", "shank_idx", "waveform_scale_uV_per_bit"):
            if key in saved:
                values[key] = saved[key]
        values["batch_options"].update(saved.get("batch_options", {}))
    probe_name = cache.value("probe_name", "Not configured in the Kilosort4 GUI")
    probe = cache.value("probe_layout")
    if PARAMETER_FILE.is_file():
        saved = json.loads(PARAMETER_FILE.read_text(encoding="utf-8"))
        probe_name = saved.get("probe_name", probe_name)
        probe = saved.get("probe", probe)
    return values, str(probe_name), probe

class ParameterWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kilosort4 Batch Parameter Settings")
        self.geometry("820x760")
        self.minsize(680, 520)
        self.values, probe_name, self.probe = load_last_values()
        self.probe_name = probe_name
        self.entries = {}
        self.option_vars = {}
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        ttk.Label(header, text=f"Current probe: {probe_name}").pack(anchor="w")
        ttk.Label(header, text=f"Settings file: {PARAMETER_FILE}").pack(anchor="w")
        ttk.Label(header, text="The probe layout is loaded from the saved batch configuration, or the last GUI settings.").pack(anchor="w")
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        canvas = tk.Canvas(body, highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas, padding=(10, 0, 10, 10))
        form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        ttk.Label(form, text="Kilosort settings").grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 8))
        row = 1
        for key in DEFAULT_SETTINGS:
            ttk.Label(form, text=key).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            entry = ttk.Entry(form)
            entry.insert(0, display_value(self.values["settings"].get(key, DEFAULT_SETTINGS[key])))
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.entries[key] = entry
            row += 1
        for key in ("data_dtype", "bad_channels", "shank_idx", "waveform_scale_uV_per_bit"):
            ttk.Label(form, text=key).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            entry = ttk.Entry(form)
            entry.insert(0, display_value(self.values[key]))
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.entries[key] = entry
            row += 1
        ttk.Label(form, text="Batch options").grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 6))
        row += 1
        for key, default in BATCH_OPTION_DEFAULTS.items():
            variable = tk.BooleanVar(value=bool(self.values["batch_options"].get(key, default)))
            ttk.Checkbutton(form, text=key, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w")
            self.option_vars[key] = variable
            row += 1
        form.columnconfigure(1, weight=1)
        footer = ttk.Frame(self, padding=10)
        footer.pack(fill="x")
        ttk.Button(footer, text="Restore Kilosort Defaults", command=self.restore_defaults).pack(side="left")
        ttk.Button(footer, text="Save and Close", command=self.save).pack(side="right")
        ttk.Button(footer, text="Cancel", command=self.destroy).pack(side="right", padx=(0, 8))

    def restore_defaults(self):
        for key, default in DEFAULT_SETTINGS.items():
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, display_value(default))
        for key, default in BATCH_OPTION_DEFAULTS.items():
            self.option_vars[key].set(default)

    def save(self):
        settings = {}
        try:
            for key, default in DEFAULT_SETTINGS.items():
                settings[key] = coerce(parse_value(self.entries[key].get()), default)
            data_dtype = str(parse_value(self.entries["data_dtype"].get()))
            bad_channels = parse_value(self.entries["bad_channels"].get())
            shank_idx = parse_value(self.entries["shank_idx"].get())
            waveform_scale = float(parse_value(self.entries["waveform_scale_uV_per_bit"].get()))
            if not math.isfinite(waveform_scale) or waveform_scale <= 0:
                raise ValueError("waveform_scale_uV_per_bit must be finite and greater than zero")
        except Exception as error:
            messagebox.showerror("Invalid Parameter Format", str(error))
            return
        payload = {
            "settings": settings,
            "data_dtype": data_dtype,
            "waveform_scale_uV_per_bit": waveform_scale,
            "bad_channels": bad_channels,
            "shank_idx": shank_idx,
            "batch_options": {key: variable.get() for key, variable in self.option_vars.items()},
            "probe_name": self.probe_name,
            "probe": self.probe,
        }
        PARAMETER_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = PARAMETER_FILE.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=json_value),
            encoding="utf-8",
        )
        temporary.replace(PARAMETER_FILE)
        messagebox.showinfo(
            "Settings Saved",
            "The parameters were saved and will be used automatically for future Kilosort batch runs.",
        )
        self.destroy()

if __name__ == "__main__":
    ParameterWindow().mainloop()
