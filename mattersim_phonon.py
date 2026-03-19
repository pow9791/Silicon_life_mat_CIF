#!/usr/bin/env python3
"""
MatterSim Phonon Check
Runs the phonon workflow on the best relaxed candidate.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Keep memory usage down by restricting threads if needed
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

print("Loading MatterSim potential...")
try:
    from mattersim.forcefield.potential import MatterSimCalculator
    from mattersim.applications.phonon import PhononWorkflow
except ImportError as e:
    print(f"Error importing mattersim: {e}")
    sys.exit(1)

# Get the best candidate
df = pd.read_csv("relaxed_energies.csv")
best = df.sort_values("energy_per_atom_eV").iloc[0]
filename = best["file"]
print(f"Best candidate: {filename} (E = {best['energy_per_atom_eV']} eV/atom)")

path = Path("relaxed_top50_cifs") / filename
if not path.exists():
    print(f"File {path} not found.")
    sys.exit(1)

print("Loading structure...")
struct = Structure.from_file(str(path))
atoms = AseAtomsAdaptor.get_atoms(struct)
atoms.calc = MatterSimCalculator()

print("Setting up Phonon workflow...")
# Use 2x2x2 supercell for phonon calculation as recommended
supercell_matrix = np.diag([2, 2, 2])

try:
    ph = PhononWorkflow(
        atoms=atoms,
        find_prim=False,
        work_dir="phonon_output",
        amplitude=0.01,
        supercell_matrix=supercell_matrix,
    )
    
    print("Running phonon calculation (this may take a few minutes)...")
    has_imag, phonons = ph.run()
    
    print("\n" + "="*50)
    print(f"PHONON RESULTS FOR {filename}")
    print(f"Has imaginary modes: {has_imag}")
    print("="*50)
    
    # Write summary
    with open("phonon_summary.txt", "w") as f:
        f.write(f"Candidate: {filename}\n")
        f.write(f"Has imaginary modes: {has_imag}\n")
        f.write(f"Supercell: 2x2x2 ({len(atoms)*8} atoms)\n")
        
except Exception as e:
    print("\nPhonon calculation failed!")
    traceback.print_exc()
