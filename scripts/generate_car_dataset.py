#!/usr/bin/env python3
"""Generate YAML dataset manifests for the Car surface dataset.

No external dependencies — pure stdlib YAML generation.
Auto-detects WSL vs Windows paths.

Usage:
    python3 generate_car_dataset.py                        # Auto-detect data root
    python3 generate_car_dataset.py --data-root /mnt/e/Car # Explicit data root
    python3 generate_car_dataset.py --dry-run              # Validate only, no write
    python3 generate_car_dataset.py --check                # Validate existing manifests
"""

import argparse
import os
import re
import sys
from pathlib import Path

SURFACE_ID = "car_part"
TRAIN_CUTOFF = 29  # groups 000-029 → train, 030+ → test

# File pattern: A{angle}_I{light}.png
FILE_RE = re.compile(r"^(A\d{3})_I(\d{2})\.(png|bmp|jpg|tif)$", re.IGNORECASE)

# Angle → position index (0-11)
ANGLES_ORDERED = [
    "A000", "A030", "A060", "A090",
    "A120", "A150", "A180", "A210",
    "A240", "A270", "A300", "A330",
]
ANGLE_TO_POS = {a: i for i, a in enumerate(ANGLES_ORDERED)}


def detect_data_root():
    """Auto-detect the Car dataset root directory."""
    candidates = [
        Path("/mnt/e/Car"),       # WSL
        Path("E:/Car"),           # Windows (native)
        Path("/e/Car"),           # Git Bash
    ]
    for c in candidates:
        if c.exists() and (c / "Good").exists():
            return c
    return None


def detect_output_dir():
    """Auto-detect the datasets/car output directory."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return repo_root / "datasets" / "car"


def yaml_str(v):
    """Naive YAML string quoting (good enough for file paths)."""
    if any(c in v for c in '":{}[]#&*!|>\'%@`,'):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return v


def scan_images(group_path: Path):
    """Scan directory, return list of (path, position, light)."""
    entries = []
    try:
        names = sorted(os.listdir(str(group_path)))
    except OSError:
        return entries
    for fname in names:
        m = FILE_RE.match(fname)
        if not m:
            continue
        angle = m.group(1).upper()
        light = int(m.group(2))
        if angle not in ANGLE_TO_POS:
            continue
        full = str((group_path / fname).resolve())
        entries.append((full, ANGLE_TO_POS[angle], light))
    return entries


def write_manifest(out: Path, entries, expected_label=None):
    """Write YAML in BasicImporter format. Returns count.

    If expected_label is not None, emits an 'expected' field for every entry.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w", encoding="utf-8") as f:
        f.write(f"surface: {yaml_str(SURFACE_ID)}\n")
        f.write("images:\n")
        for path_item, pos, light in entries:
            f.write(f'  - path: {yaml_str(path_item)}\n')
            f.write(f'    position: {pos}\n')
            f.write(f'    light: {light}\n')
            if expected_label is not None:
                f.write(f'    expected: {expected_label}\n')
    return len(entries)


def validate_manifest(path: Path):
    """Check an existing manifest for consistency."""
    issues = []
    if not path.exists():
        issues.append(f"MISSING: {path}")
        return issues

    with open(str(path), encoding="utf-8") as f:
        content = f.read()

    if not content.startswith("surface:"):
        issues.append(f"{path.name}: missing 'surface:' header")
    if "images:" not in content.split("\n", 2)[:3]:
        issues.append(f"{path.name}: missing 'images:' key")

    # Count path/position/light triples
    path_count = content.count("\n  - path:")
    pos_count = content.count("\n    position:")
    light_count = content.count("\n    light:")
    if path_count != pos_count or path_count != light_count:
        issues.append(f"{path.name}: mismatched counts (path={path_count}, pos={pos_count}, light={light_count})")

    # Count expected labels (present only in test manifest files)
    expected_count = content.count("\n    expected:")
    if expected_count > 0 and expected_count != path_count:
        issues.append(f"{path.name}: expected count ({expected_count}) != path count ({path_count})")

    # Check for backslashes in paths
    if "\\" in content and "/" not in content:
        pass  # pure Windows paths allowed
    elif "\\" in content:
        issues.append(f"{path.name}: contains mixed path separators")

    # Verify all referenced files exist (sample check on first 10)
    path_lines = [l for l in content.split("\n") if l.strip().startswith("- path:")]
    missing_files = 0
    for line in path_lines[:20]:  # Sample check first 20
        m = re.search(r'path:\s*"?(.+?)"?(?:\s*$)', line)
        if m:
            fpath = Path(m.group(1))
            if not fpath.exists():
                missing_files += 1

    if missing_files > 0:
        issues.append(f"{path.name}: {missing_files}/{min(20, len(path_lines))} sampled paths point to missing files")

    if not issues:
        return []
    return issues


def main():
    parser = argparse.ArgumentParser(description="Generate Car dataset YAML manifests")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to Car dataset root (default: auto-detect)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for YAML manifests (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report without writing files")
    parser.add_argument("--check", action="store_true",
                        help="Validate existing manifests only, do not regenerate")
    parser.add_argument("--wsl", action="store_true",
                        help="Force WSL-style paths (/mnt/e/...) in output")
    args = parser.parse_args()

    # ── Resolve paths ──
    data_root = Path(args.data_root) if args.data_root else detect_data_root()
    if data_root is None:
        print("ERROR: Cannot auto-detect Car dataset root.")
        print("  Tried: /mnt/e/Car (WSL), E:/Car (Windows), /e/Car (Git Bash)")
        print("  Specify with --data-root PATH")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else detect_output_dir()
    use_wsl = args.wsl

    is_wsl_platform = str(data_root).startswith('/mnt/')
    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")
    print(f"Platform:    {'WSL' if is_wsl_platform else 'Windows'}{' (--wsl forced)' if use_wsl and not is_wsl_platform else ''}")

    if not data_root.exists():
        print(f"\nERROR: Data root does not exist: {data_root}")
        sys.exit(1)
    if not (data_root / "Good").exists():
        print(f"\nERROR: 'Good' directory not found under: {data_root}")
        sys.exit(1)

    # ── Check mode ──
    if args.check:
        print("\n=== Validating existing manifests ===\n")
        all_ok = True
        for fname in ["car_train.yaml", "car_test_ok.yaml", "car_test_ng.yaml", "car_test_all.yaml"]:
            issues = validate_manifest(output_dir / fname)
            if issues:
                all_ok = False
                for i in issues:
                    print(f"  FAIL: {i}")
            else:
                print(f"  PASS: {fname}")
        sys.exit(0 if all_ok else 1)

    # ── Scan Good ──
    good_dir = data_root / "Good"
    good_groups = sorted(
        d.name for d in good_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not good_groups:
        print(f"\nERROR: No numeric group directories found under: {good_dir}")
        sys.exit(1)

    print(f"\nGood groups: {len(good_groups)} ({good_groups[0]} ~ {good_groups[-1]})")

    train = []
    test_ok = []

    for g in good_groups:
        gid = int(g)
        imgs = scan_images(good_dir / g)
        expected = 12 * 10  # 12 angles × 10 lights
        status = "OK" if len(imgs) == expected else f"WARN: expected {expected}"
        print(f"  Good/{g}: {len(imgs)} images ({status})")
        if gid <= TRAIN_CUTOFF:
            train.extend(imgs)
        else:
            test_ok.extend(imgs)

    # ── Scan NG ──
    ng_dir = data_root / "NG"
    ng_groups = sorted(
        d.name for d in ng_dir.iterdir()
        if d.is_dir()
    )
    if not ng_groups:
        print("\nWARNING: No NG groups found (this may be intentional)")

    print(f"\nNG groups: {len(ng_groups)}")

    ng_all = []
    for g in ng_groups:
        imgs = scan_images(ng_dir / g)
        print(f"  NG/{g}: {len(imgs)} images")
        ng_all.extend(imgs)

    # ── Normalize paths ──
    def to_posix(entries):
        return [(p.replace("\\", "/"), pos, lt) for p, pos, lt in entries]

    def to_wsl(entries):
        """Convert Windows paths (E:/Car/...) to WSL paths (/mnt/e/Car/...)."""
        result = []
        for p, pos, lt in entries:
            p = p.replace("\\", "/")
            # E:/Car/... → /mnt/e/Car/...    E:/ → /mnt/e/
            if len(p) >= 2 and p[1] == ':':
                drive_letter = p[0].lower()
                p = f"/mnt/{drive_letter}{p[2:]}"
            result.append((p, pos, lt))
        return result

    if use_wsl:
        train = to_wsl(train)
        test_ok = to_wsl(test_ok)
        ng_all = to_wsl(ng_all)
    else:
        train = to_posix(train)
        test_ok = to_posix(test_ok)
        ng_all = to_posix(ng_all)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Split summary:")
    print(f"  Train (Good 000-{TRAIN_CUTOFF:03d}): {len(train)} images, {len(train)//120} groups")
    print(f"  Test OK (Good {TRAIN_CUTOFF+1:03d}+): {len(test_ok)} images, {len(test_ok)//120} groups")
    print(f"  Test NG (all):                    {len(ng_all)} images, {len(ng_all)//120} groups")
    print(f"  Test ALL (OK+NG):                 {len(test_ok) + len(ng_all)} images")

    if args.dry_run:
        print(f"\n--dry-run: no files written.")
        # Validate a few random paths
        all_entries = train + test_ok + ng_all
        missing = 0
        for p, _, _ in all_entries[:50]:
            if not Path(p).exists():
                missing += 1
        if missing:
            print(f"WARNING: {missing}/{min(50, len(all_entries))} sampled paths not found on disk")
        else:
            print("Sample path check: OK")
        return

    # ── Write ──
    print(f"\nWriting manifests to: {output_dir}")
    manifests = [
        ("car_train.yaml", train, "TRAIN", None),        # train 不设 expected
        ("car_test_ok.yaml", test_ok, "TEST_OK", "OK"),   # 正常样本 → expected: OK
        ("car_test_ng.yaml", ng_all, "TEST_NG", "NG"),    # NG 样本 → expected: NG
        ("car_test_all.yaml", test_ok + ng_all, "TEST_ALL", None),  # 混合测试不设 expected (从 source 继承)
    ]

    for fname, entries, label, expected_label in manifests:
        n = write_manifest(output_dir / fname, entries, expected_label=expected_label)
        print(f"  {label:12s} → {fname:25s} ({n} images)")

    # ── Quick validation ──
    print(f"\nPost-write validation:")
    all_ok = True
    for fname, _, label, _ in manifests:
        issues = validate_manifest(output_dir / fname)
        if issues:
            all_ok = False
            for i in issues:
                print(f"  FAIL: {i}")
        else:
            print(f"  PASS: {fname}")

    if all_ok:
        print(f"\nDone. All manifests generated successfully.")
    else:
        print(f"\nDone with warnings.")
        sys.exit(1)


if __name__ == "__main__":
    main()
