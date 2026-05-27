"""Command-line filtering of CREST-style multi-conformer XYZ files.

Reads an XYZ file produced by CREST (e.g. ``crest_conformers.xyz``) where the
comment line of each block contains the conformer energy in Hartree, runs
``ConformerEnsemble.filter`` with chemtrayzer-compatible options, and writes the
deduplicated set back out as an XYZ file (or to stdout).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from . import ConfFilterOptions, ConformerEnsemble


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crest-filter",
        description=(
            "Filter a CREST-style multi-conformer XYZ file by RMSD, rotational "
            "constants, energy window, and cumulative Boltzmann weight."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input XYZ file (concatenated blocks).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write filtered XYZ here (default: stdout).",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.125,
        help="RMSD threshold [Angstrom] (default: 0.125).",
    )
    parser.add_argument(
        "--rot-threshold",
        type=float,
        default=1.0,
        help="Rotational-constant threshold [%%] (default: 1.0).",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.00038,
        help="Energy difference threshold [Hartree] (default: 0.00038).",
    )
    parser.add_argument(
        "--energy-window",
        type=float,
        default=math.inf,
        help=(
            "Drop conformers with E - E_min > energy_window [Hartree] "
            "(default: inf)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1500.0,
        help="Temperature [K] for Boltzmann weighting (default: 1500).",
    )
    parser.add_argument(
        "--cum-boltzmann",
        dest="cum_boltzmann_threshold",
        type=float,
        default=1.0,
        help=(
            "Cumulative Boltzmann weight threshold in (0, 1] (default: 1.0, "
            "i.e. keep all)."
        ),
    )
    parser.add_argument(
        "--mass-weighted",
        action="store_true",
        help="Use mass-weighted RMSD.",
    )
    parser.add_argument(
        "--no-rigid-transform",
        dest="rigid_transform",
        action="store_false",
        default=True,
        help="Disable rigid rotation alignment during RMSD.",
    )
    parser.add_argument(
        "--mirror-check",
        action="store_true",
        help="Also compare each conformer against the mirror image of others.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress the summary line on stderr.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        ensemble = ConformerEnsemble.from_xyz_file(str(args.input))
    except FileNotFoundError:
        print(f"crest-filter: input not found: {args.input}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(
            f"crest-filter: failed to read {args.input}: {exc}",
            file=sys.stderr,
        )
        return 2

    opts = ConfFilterOptions(
        rmsd_threshold=args.rmsd_threshold,
        rot_threshold=args.rot_threshold,
        mass_weighted=args.mass_weighted,
        rigid_transform=args.rigid_transform,
        energy_threshold=args.energy_threshold,
        mirror_check=args.mirror_check,
        temperature=args.temperature,
        cum_boltzmann_threshold=args.cum_boltzmann_threshold,
        energy_window=args.energy_window,
    )

    n_in = len(ensemble)
    try:
        filtered = ensemble.filter(opts)
    except ValueError as exc:
        print(f"crest-filter: filter failed: {exc}", file=sys.stderr)
        return 1
    n_out = len(filtered)

    text = filtered.xyz_str()
    if args.output is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        args.output.write_text(text, encoding="utf-8")

    if not args.quiet:
        print(
            f"crest-filter: kept {n_out}/{n_in} conformers "
            f"(removed {n_in - n_out}).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
