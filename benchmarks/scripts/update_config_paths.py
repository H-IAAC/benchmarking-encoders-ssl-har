"""
update_config_paths.py
======================
Rewrite all data_path values in base_configs YAML files to match the current
repository location.

The data_path values follow the pattern:
    data_path: "<old_repo_root>/shared_data/<suffix>"

This script replaces <old_repo_root> with the actual repo root, detected in
this order:
  1. --root flag (explicit override)
  2. Git: `git rev-parse --show-toplevel`
  3. Filesystem walk: climb from this script's location until a directory
     containing `benchmarks/base_configs` is found.

Usage
-----
    python update_config_paths.py             # auto-detect repo root
    python update_config_paths.py --root /my/path/benchmarking-encoders-ssl-har
    python update_config_paths.py --dry-run   # preview only, no changes written
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


# Landmark that must exist directly under the repo root.
_LANDMARK = Path("benchmarks") / "base_configs"


def find_repo_root_git() -> Path | None:
    """Return the git repo root, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def find_repo_root_filesystem() -> Path | None:
    """Walk up from this script's directory looking for _LANDMARK."""
    current = Path(__file__).resolve().parent
    while True:
        if (current / _LANDMARK).exists():
            return current
        parent = current.parent
        if parent == current:  # reached filesystem root
            return None
        current = parent


def find_repo_root() -> Path | None:
    return find_repo_root_git() or find_repo_root_filesystem()


def update_file(path: Path, new_root: Path, dry_run: bool) -> bool:
    """Return True if the file was (or would be) modified."""
    text = path.read_text()

    # Match:  data_path: "<anything>/shared_data/<suffix>"
    #      or processed_data_dir: "<anything>/shared_data/<suffix>"
    # Groups:
    #   1 = key + opening quote,  e.g. 'processed_data_dir: "'
    #   2 = key name only         (used for alternation, not in replacement)
    #   3 = suffix after shared_data/  (no quotes)
    pattern = re.compile(
        r'((data_path|processed_data_dir):\s*")[^"]*/shared_data/([^"]*)"'
    )

    new_text, count = pattern.subn(
        lambda m: f'{m.group(1)}{new_root}/shared_data/{m.group(3)}"',
        text,
    )

    if count == 0 or new_text == text:
        return False

    if dry_run:
        for old_line, new_line in zip(text.splitlines(), new_text.splitlines()):
            if old_line != new_line:
                print(f"  - {old_line.strip()}")
                print(f"  + {new_line.strip()}")
        return True

    path.write_text(new_text)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        help="Repo root path (default: auto-detect via git or filesystem walk)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing files",
    )
    args = parser.parse_args()

    if args.root:
        repo_root = Path(args.root).resolve()
    else:
        repo_root = find_repo_root()
        if repo_root is None:
            print(
                "ERROR: Could not detect the repo root automatically.\n"
                "  Tried: git rev-parse --show-toplevel\n"
                f"  Tried: walking up from {Path(__file__).resolve()} "
                f"looking for '{_LANDMARK}'\n"
                "Use --root to specify the path explicitly."
            )
            sys.exit(1)

    configs_dir = repo_root / _LANDMARK
    if not configs_dir.exists():
        print(f"ERROR: base_configs not found at {configs_dir}")
        sys.exit(1)

    yaml_files = list(configs_dir.rglob("*.yaml"))
    print(f"Repo root : {repo_root}")
    print(f"Scanning  : {configs_dir}  ({len(yaml_files)} YAML files)")
    if args.dry_run:
        print("Dry run   : no files will be written\n")

    changed = 0
    for f in sorted(yaml_files):
        if update_file(f, repo_root, dry_run=args.dry_run):
            rel = f.relative_to(repo_root)
            print(f"{'[dry]' if args.dry_run else '[updated]':10s} {rel}")
            changed += 1

    print(
        f"\n{'Would update' if args.dry_run else 'Updated'} {changed}/{len(yaml_files)} files."
    )


if __name__ == "__main__":
    main()
