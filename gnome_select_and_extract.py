#!/usr/bin/env python3
"""
GNoME Silicon-Life CIF Pipeline
================================
Select the top 50 silicon-rich molecular/flexible (non-crystal) candidates
from the GNoME stable_materials_summary.csv dataset and extract their CIF
files from the by_id.zip archive.

Scoring criteria (protein-analogue suitability for Si-life at 40-100 °C):
 - Low decomposition energy (near convex hull) → thermodynamic stability
 - Structural complexity (more sites) → framework/scaffold potential
 - Zero or low-dimensional motifs (0D/Molecular/1D) → flexible, solvent-accessible behavior
 - Small-to-moderate band gap → electronically active scaffolds
 - Preference for core crustal elements (Si,Fe,Al,Ca,Na,K,Mg,Ti)
"""

import argparse
import csv
import os
import re
import zipfile
from pathlib import Path

import pandas as pd


# ── Element sets ───────────────────────────────────────────────────────
# Extremely abundant crustal elements (the building blocks)
CORE = {"Si", "Al", "Fe", "Ca", "Na", "K", "Mg", "Ti"}
CORE_METAL = CORE - {"Si"}

# Plausible trace elements for catalytic centers, but cannot dominate
EXTENDED = CORE | {
    "Mn", "Cr", "V", "Ni", "Co", "Zn", "Cu", "Ba", "Sr", "Li", "B"
}
# Hard-exclude: Noble Gases, Rare Earths, Noble/Heavy Metals.
# ALLOW Halogens and Chalcogens (mist/solvent compatible).
# NO Rare Earths (Tb, Dy, Ho, La, Ac, Y, etc.)
# NO Noble/Platinum Group Metals (Os, Ru, Ir, Pt, Pd, Rh, Au, Ag)
# NO toxic/rare heavy metals (Pb, Tl, Bi, Hg, Cd)
ALLOWED_ALL = EXTENDED | {"S", "P", "C", "N", "O", "Se", "Te", "F", "Cl", "Br", "I"}



def parse_elements(s: str) -> list[str]:
    """Parse the Elements column which looks like "['Si', 'Fe']"."""
    s = str(s).strip()
    if s.startswith("["):
        return [x.strip().strip("'\"") for x in s.strip("[]").split(",")]
    if "-" in s:
        return [x for x in s.split("-") if x]
    return re.findall(r"[A-Z][a-z]?", s)


def pick_decomp_col(df: pd.DataFrame) -> str:
    candidates = [
        "Decomposition Energy Per Atom All",
        "Decomposition Energy Per Atom",
        "Decomposition Energy Per Atom MP OQMD",
        "Decomposition Energy Per Atom MP",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No decomposition-energy column found.")


def score_row(row, decomp_col: str) -> float:
    """Score each candidate for silicon-life scaffold suitability."""
    score = 0.0

    # ── Thermodynamic stability: lower hull distance → better ──────
    de = float(row.get(decomp_col, 1e9))
    if de <= 0.0:       # ON the hull
        score += 10.0
    elif de <= 0.005:
        score += 8.0
    elif de <= 0.01:
        score += 6.0
    elif de <= 0.02:
        score += 4.0
    elif de <= 0.05:
        score += 2.0

    # ── Structural complexity: more sites → framework potential for large proteins ────
    nsites = float(row.get("NSites", 0))
    # Cap raised significantly (e.g., from 40 to 200) to reward very large, complex scaffolds
    score += min(nsites, 200) * 0.15

    # ── Dimensionality: strongly prefer molecules/1D for flexible non-crystals ─────
    dim = str(row.get("Dimensionality Cheon", "")).strip()
    if dim in {"0D", "molecule"}:
        score += 8.0
    elif dim in {"intercalated ion", "1", "1D"}:
        score += 4.0
    elif dim in {"2", "2D"}:
        score += 1.0
    elif dim in {"3", "3D"}:
        score -= 5.0  # Penalize rigid 3D crystals heavily

    # ── Electronic activity: small-to-moderate band gap ────────────
    bg = row.get("Bandgap", None)
    try:
        bg = float(bg)
        if 0.1 <= bg <= 2.5:
            score += 2.0
        elif 0.0 < bg < 0.1:
            score += 1.0
    except (TypeError, ValueError):
        pass

    # ── Element preference: MUST scale to 100s of billions of lifeforms ───────────
    # We heavily reward absolute core elements (Si, Al, Fe, O, C, N).
    els = set(row.get("_els", []))
    core_frac = len(els & CORE) / max(len(els), 1)
    
    # Trace elements (in EXTENDED but not CORE) are allowed for special enzymes.
    # We give a small bonus for having them, but penalize heavily if they dominate the structure too much.
    trace_frac = len(els & (EXTENDED - CORE)) / max(len(els), 1)
    
    score += core_frac * 6.0  # Double reward for ultra-abundant elements
    
    if trace_frac > 0:
        if trace_frac <= 0.25:
            score += 2.0  # Encourage trace metals for specialized enzymatic centers
        else:
            score -= (trace_frac * 4.0) # Penalize only if trace elements dominate the structure

    # ── Silicon richness: reward high Si fraction ──────────────────
    formula = str(row.get("Reduced Formula", ""))
    si_atoms = sum(int(n) if n else 1 for n in re.findall(r"Si(\d*)", formula))
    total_atoms = sum(int(n) if n else 1 for n in re.findall(r"[A-Z][a-z]?(\d*)", formula))
    if total_atoms > 0:
        si_frac = si_atoms / total_atoms
        score += si_frac * 4.0

    return round(score, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--cif_zip", required=True, help="by_id.zip from GNoME")
    ap.add_argument("--out_csv", default="top50_gnome_candidates.csv")
    ap.add_argument("--out_dir", default="gnome_top50_cifs")
    ap.add_argument("--max_hull", type=float, default=0.05,
                    help="Max decomposition energy per atom (eV)")
    ap.add_argument("--n_top", type=int, default=50)
    args = ap.parse_args()

    # ── Load & initial filter ──────────────────────────────────────
    print("Loading CSV ...")
    df = pd.read_csv(args.summary_csv)
    decomp_col = pick_decomp_col(df)
    print(f"  {len(df)} total materials, using '{decomp_col}'")

    df["_els"] = df["Elements"].apply(parse_elements)

    # Must contain Si
    df = df[df["_els"].apply(lambda els: "Si" in els)]
    print(f"  {len(df)} contain Si")

    # STRICT INCLUSION: Only crustal + trace allowed. No rare earths/noble metals!
    df = df[df["_els"].apply(lambda els: set(els).issubset(ALLOWED_ALL))]
    print(f"  {len(df)} after strictly enforcing crustal/trace-only elements")

    # Prefer core elements (reward high fractions of Si, Al, Fe, Ca, Na, K, Mg, Ti)
    df["_core_frac"] = df["_els"].apply(
        lambda els: len(set(els) & CORE) / max(len(els), 1)
    )

    # Hull distance filter
    df[decomp_col] = pd.to_numeric(df[decomp_col], errors="coerce")
    # Using a slightly looser hull to guarantee 50 complex candidates with strict crustal elements
    max_hull = max(args.max_hull, 0.1) 
    df = df[df[decomp_col] <= max_hull]
    print(f"  {len(df)} with hull distance <= {max_hull} eV/atom")

    # ── Score & rank ───────────────────────────────────────────────
    df["_score"] = df.apply(lambda r: score_row(r, decomp_col), axis=1)
    df = df.sort_values(
        ["_score", decomp_col, "NSites"],
        ascending=[False, True, False],
    )

    top = df.head(args.n_top).copy()
    print(f"\nTop {len(top)} candidates selected.")

    # ── Save ranked table ──────────────────────────────────────────
    out_cols = [
        "MaterialId", "Reduced Formula", "Composition", "Elements",
        "NSites", "Volume", "Density", "Space Group", "Crystal System",
        "Formation Energy Per Atom", decomp_col,
        "Dimensionality Cheon", "Bandgap", "_score",
    ]
    out_cols = [c for c in out_cols if c in top.columns]
    top[out_cols].to_csv(args.out_csv, index=False)
    print(f"Saved ranked table → {args.out_csv}")

    # ── Extract CIFs ───────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.cif_zip} ...")
    with zipfile.ZipFile(args.cif_zip) as zf:
        members = zf.namelist()
        # Build lookup: basename → full path in zip
        name_map = {}
        for m in members:
            base = os.path.basename(m)
            name_map[base] = m

        extracted = 0
        missing = []
        for _, row in top.iterrows():
            mid = str(row["MaterialId"])
            rform = str(row.get("Reduced Formula", "unknown")).replace("/", "_")

            # Try matching by MaterialId in filename
            match = None
            for base, full_path in name_map.items():
                if mid in base:
                    match = full_path
                    break

            if match is None:
                missing.append(f"{mid} ({rform})")
                continue

            target_name = f"{mid}_{rform}.cif"
            with zf.open(match) as src, open(out_dir / target_name, "wb") as dst:
                dst.write(src.read())
            extracted += 1

        print(f"Extracted {extracted} CIF files → {args.out_dir}/")
        if missing:
            print(f"Could not find CIFs for {len(missing)} entries:")
            for m in missing:
                print(f"  [WARN] {m}")


if __name__ == "__main__":
    main()
