#!/usr/bin/env python3

"""
make_star_links.py

Scan BLOeM DR4.0 combined FITS files across FIELD0..FIELD8 and create a directory
structure of symlinks grouped by star name.

Input layout (example):
/Users/roeyovadia/Documents/Data/BLOeM_DR4.0_Combined/FIELD{field_num}/FITS/{star_name}_{epoch_num}_Combined.fits

Output layout:
/out_dir/{star_name}/<original_filename>.fits  (symlink to the original)

Usage:
  python make_star_links.py \
      --root "/Users/roeyovadia/Documents/Data/BLOeM_DR4.0_Combined" \
      --out-dir "/path/to/out_dir"

Options:
  --force          Overwrite existing files/links in the out_dir.
  --relative       Make symlinks relative instead of absolute.
  --dry-run        Show what would be done without creating anything.
  --fields 0 1 2   Restrict to specific field numbers (default: 0..8).
  --pattern        Custom filename regex (advanced).
  --verbose        Print progress.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

DEFAULT_REGEX = r"(?P<star>.+)_(?P<epoch>\d+)_Combined\.fits$"
def load_star_list(path: Path) -> set[str]:
    stars: set[str] = set()
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        stars.add(s)
    return stars

def iter_fits_files(root: Path, fields: Iterable[int]) -> Iterable[Path]:
    for f in fields:
        fits_dir = root / f"FIELD{f}" / "FITS"
        if not fits_dir.exists():
            continue
        yield from fits_dir.glob("*.fits")

def parse_star_and_epoch(path: Path, pattern: str) -> tuple[str, str] | None:
    m = re.search(pattern, path.name)
    if not m:
        return None
    return m.group("star"), m.group("epoch")

def ensure_dir(p: Path, dry_run: bool, verbose: bool):
    if p.exists():
        return
    if verbose:
        print(f"[mkdir] {p}")
    if not dry_run:
        p.mkdir(parents=True, exist_ok=True)

def make_symlink(src: Path, dst: Path, relative: bool, force: bool, dry_run: bool, verbose: bool):
    # Ensure parent exists
    ensure_dir(dst.parent, dry_run, verbose)

    # Compute link target
    target = src
    if relative:
        try:
            target = Path(Path.relpath(src, start=dst.parent))
        except Exception:
            # Fall back to absolute if relative fails
            target = src

    # Handle existing destination
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            existing_target = dst.readlink()
            # Normalize for comparison
            if existing_target == target or (not relative and existing_target == src):
                if verbose:
                    print(f"[skip] link already correct: {dst} -> {existing_target}")
                return
        if not force:
            if verbose:
                print(f"[skip] exists (use --force to overwrite): {dst}")
            return
        # Remove and overwrite
        if verbose:
            print(f"[rm] existing {'link' if dst.is_symlink() else 'file'}: {dst}")
        if not dry_run:
            dst.unlink()

    if verbose:
        print(f"[ln -s] {dst} -> {target}")
    if not dry_run:
        dst.symlink_to(target)

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Group BLOeM DR4.0 FITS by star into symlinked folders.")
    p.add_argument("--root", type=Path, required=True,
                   help="Root directory containing FIELD0..FIELD8 (e.g., /Users/roeyovadia/Documents/Data/BLOeM_DR4.0_Combined)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory to create grouped symlinks in.")
    p.add_argument(
        "--star-list",
        type=Path,
        default=None,
        help="Optional path to a text file with one star_name per line. "
             "Lines starting with # and empty lines are ignored. "
             "If given, only these stars will be linked."
    )
    p.add_argument("--fields", type=int, nargs="+", default=list(range(0, 9)),
                   help="Field numbers to include (default: 0..8).")
    p.add_argument("--pattern", type=str, default=DEFAULT_REGEX,
                   help=f"Regex with groups 'star' and 'epoch' (default: {DEFAULT_REGEX})")
    p.add_argument("--relative", action="store_true", help="Create relative symlinks instead of absolute.")
    p.add_argument("--force", action="store_true", help="Overwrite existing links/files at destination.")
    p.add_argument("--dry-run", action="store_true", help="Only print actions without creating anything.")
    p.add_argument("--verbose", action="store_true", help="Print progress.")
    args = p.parse_args(argv)

    root: Path = args.root.expanduser().resolve()
    out_dir: Path = args.out_dir.expanduser().resolve()
    fields: List[int] = args.fields
    pattern: str = args.pattern

    star_filter: set[str] | None = None
    if args.star_list is not None:
        star_list_path = args.star_list.expanduser().resolve()
        if not star_list_path.exists():
            print(f"[ERR] Star list file not found: {star_list_path}", file=sys.stderr)
            return 2
        star_filter = load_star_list(star_list_path)
        if args.verbose:
            print(f"[info] star_list={star_list_path} n_stars={len(star_filter)}")

    if not root.exists():
        print(f"[ERR] Root not found: {root}", file=sys.stderr)
        return 2

    if args.verbose:
        print(f"[info] root={root}")
        print(f"[info] out_dir={out_dir}")
        print(f"[info] fields={fields}")
        print(f"[info] pattern={pattern}")

    # Create base out_dir
    ensure_dir(out_dir, args.dry_run, args.verbose)

    total_seen = 0
    total_linked = 0
    total_skipped = 0
    total_failed = 0

    for fp in iter_fits_files(root, fields):
        total_seen += 1
        parsed = parse_star_and_epoch(fp, pattern)
        if not parsed:
            total_skipped += 1
            if args.verbose:
                print(f"[skip:nomatch] {fp.name}")
            continue
        star_name, epoch = parsed
        if star_filter is not None and star_name not in star_filter:
            total_skipped += 1
            if args.verbose:
                print(f"[skip:notinlist] {star_name} ({fp.name})")
            continue

        # Keep original filename
        dst_dir = out_dir / star_name
        dst = dst_dir / fp.name
        try:
            make_symlink(fp, dst, args.relative, args.force, args.dry_run, args.verbose)
            total_linked += 1
        except Exception as e:
            total_failed += 1
            print(f"[ERR] {fp} -> {dst}: {e}", file=sys.stderr)

    if args.verbose or True:
        print(f"[done] seen={total_seen} linked={total_linked} skipped={total_skipped} failed={total_failed}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
