#!/usr/bin/env python3
"""
MatterSim relaxation for GNoME silicon-life candidates.
Relaxes each structure individually (memory-safe for 8 GB RAM machines).
Saves relaxed CIFs and energies.
"""

import csv
import gc
import os
import sys
import traceback
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

IN_DIR = Path("gnome_top50_cifs")
OUT_DIR = Path("relaxed_top50_cifs")
OUT_DIR.mkdir(exist_ok=True)

paths = sorted(IN_DIR.glob("*.cif"))
print(f"Found {len(paths)} CIF files to relax")

# ── Imports for ASE optimization (ASE 3.27+ moved filters) ─────────
from ase.optimize import FIRE
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter

# ── Lazy-load MatterSim (downloads model checkpoint on first use) ──
print("Loading MatterSim potential (5M model) ...")
try:
    from mattersim.forcefield.potential import Potential, MatterSimCalculator
    potential = Potential.from_checkpoint('mattersim-v1.0.0-5M.pth')
    print("Potential 5M loaded.")
except Exception as e:
    print(f"ERROR loading MatterSim: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Relax each structure individually ──────────────────────────────
rows = []
for i, path in enumerate(paths):
    name = path.stem
    print(f"\n[{i+1}/{len(paths)}] {name}")

    try:
        struct = Structure.from_file(str(path))
        atoms = AseAtomsAdaptor.get_atoms(struct)

        # Attach MatterSim calculator
        atoms.calc = MatterSimCalculator(potential=potential)

        # Use ASE's built-in optimizer for memory efficiency

        ecf = ExpCellFilter(atoms)
        opt = FIRE(ecf, logfile=None)
        opt.run(fmax=0.05, steps=200)  # relaxed fmax for speed

        energy = atoms.get_potential_energy()
        energy_per_atom = energy / len(atoms)

        # Save relaxed structure
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        out_path = OUT_DIR / f"{name}.cif"
        final_struct.to(filename=str(out_path))

        rows.append({
            "file": path.name,
            "formula": final_struct.composition.reduced_formula,
            "n_atoms": len(final_struct),
            "relaxed_energy_eV": round(energy, 4),
            "energy_per_atom_eV": round(energy_per_atom, 4),
            "volume_A3": round(final_struct.volume, 2),
            "status": "OK",
        })
        print(f"  E = {energy:.4f} eV ({energy_per_atom:.4f} eV/atom), V = {final_struct.volume:.1f} Å³")

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        rows.append({
            "file": path.name,
            "formula": "?",
            "n_atoms": 0,
            "relaxed_energy_eV": None,
            "energy_per_atom_eV": None,
            "volume_A3": None,
            "status": f"FAIL: {e}",
        })

    # Force garbage collection to stay within 8 GB
    gc.collect()

# ── Save summary ──────────────────────────────────────────────────
with open("relaxed_energies.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file", "formula", "n_atoms", "relaxed_energy_eV",
        "energy_per_atom_eV", "volume_A3", "status",
    ])
    writer.writeheader()
    writer.writerows(rows)

ok = sum(1 for r in rows if r["status"] == "OK")
print(f"\n{'='*60}")
print(f"Relaxation complete: {ok}/{len(rows)} succeeded")
print(f"Relaxed structures → {OUT_DIR}/")
print(f"Energy summary     → relaxed_energies.csv")
