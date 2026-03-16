# symmetrize_lens.py
"""
Symmetrize "half-lens" / partial circular-lens multiphysics datasets into a full-lens dataset
by reflecting points across one or more coordinate planes.

Symmetry selection:
- X symmetry: reflect across YZ plane  -> (x, y, z) -> (-x, y, z)
- Y symmetry: reflect across XZ plane  -> (x, y, z) -> (x, -y, z)
- Z symmetry: reflect across XY plane  -> (x, y, z) -> (x, y, -z)

Vector handling:
Any detected vector components are transformed consistently with the reflection:
- Mirror in X: x flips sign AND any vector X-component flips sign.
- Mirror in Y: y flips sign AND any vector Y-component flips sign.
- Mirror in Z: z flips sign AND any vector Z-component flips sign.
Scalars remain unchanged.

Supported inputs:
- Optional leading comments / blank lines.
- Optional header line with column names.
- Delimiters: comma, tab, or arbitrary whitespace.
- If header exists: attempt to detect x/y/z columns (case-insensitive, common synonyms).
- If no header: first 3 numeric columns are assumed x,y,z.

Output:
- Preserves initial comment/blank preamble.
- Preserves header if present.
- Preserves delimiter if comma/tab is detected; otherwise outputs tab-delimited.
- Numeric formatting uses scientific notation with 15 digits after decimal.

Important:
- By default, NO de-duplication is performed. This guarantees output row count equals:
    input_rows * number_of_reflection_combinations
  If you want to remove duplicates on symmetry planes, set --dedupe-tol to a small positive value.

Demo:
    python symmetrize_lens.py --demo
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


COMMENT_PREFIXES = ("#", "//", ";", "%")


@dataclass(frozen=True)
class ParsedFile:
    path: Path
    preamble_lines: List[str]
    header_line: Optional[str]
    delimiter: Optional[str]
    columns: Optional[List[str]]
    data: np.ndarray  # (n_rows, n_cols)


@dataclass(frozen=True)
class ColumnModel:
    x_idx: int
    y_idx: int
    z_idx: int
    vector_groups: List[Tuple[int, int, int]]


def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return any(s.startswith(p) for p in COMMENT_PREFIXES)


def _detect_delimiter(sample_lines: Sequence[str]) -> Optional[str]:
    for line in sample_lines:
        s = line.strip()
        if not s or _is_comment_or_blank(s):
            continue
        if "," in s:
            return ","
        if "\t" in s:
            return "\t"
        return None
    return None


def _split_fields(line: str, delimiter: Optional[str]) -> List[str]:
    s = line.strip()
    if delimiter == ",":
        return [f.strip() for f in s.split(",")]
    if delimiter == "\t":
        return [f.strip() for f in s.split("\t")]
    return s.split()


_FLOAT_RE = re.compile(
    r"""
    ^[+-]?
    (?:
        (?:\d+\.\d*)|(?:\d*\.\d+)|(?:\d+)
    )
    (?:[eE][+-]?\d+)?$
    """,
    re.VERBOSE,
)


def _is_float_token(tok: str) -> bool:
    return bool(_FLOAT_RE.match(tok.strip()))


def _line_is_numeric_row(fields: Sequence[str]) -> bool:
    if len(fields) < 3:
        return False
    return all(_is_float_token(f) for f in fields[:3])


def parse_file(path: Path) -> ParsedFile:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    preamble: List[str] = []
    i = 0
    while i < len(raw_lines) and _is_comment_or_blank(raw_lines[i]):
        preamble.append(raw_lines[i])
        i += 1

    delimiter = _detect_delimiter(raw_lines[i : min(len(raw_lines), i + 20)])

    header_line: Optional[str] = None
    columns: Optional[List[str]] = None

    if i < len(raw_lines):
        fields = _split_fields(raw_lines[i], delimiter)
        has_alpha = any(re.search(r"[A-Za-z]", f) for f in fields)
        numeric_like = _line_is_numeric_row(fields)
        if has_alpha and not numeric_like:
            header_line = raw_lines[i]
            columns = fields
            i += 1

    data_rows: List[List[float]] = []
    widths: Optional[int] = None

    for line_no, line in enumerate(raw_lines[i:], start=i + 1):
        if not line.strip():
            continue
        if _is_comment_or_blank(line):
            continue

        fields = _split_fields(line, delimiter)
        if widths is None:
            widths = len(fields)

        if len(fields) != widths:
            raise ValueError(
                f"{path}: line {line_no} has {len(fields)} columns; expected {widths}. "
                f"Offending line: {line!r}"
            )

        if not all(_is_float_token(f) for f in fields):
            raise ValueError(
                f"{path}: line {line_no} contains non-numeric tokens in data section. "
                f"Offending line: {line!r}"
            )

        data_rows.append([float(f) for f in fields])

    if not data_rows:
        raise ValueError(f"{path}: no numeric data rows found.")

    data = np.asarray(data_rows, dtype=np.float64)
    return ParsedFile(
        path=path,
        preamble_lines=preamble,
        header_line=header_line,
        delimiter=delimiter,
        columns=columns,
        data=data,
    )


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def detect_columns(parsed: ParsedFile) -> ColumnModel:
    n_cols = parsed.data.shape[1]

    if parsed.columns is None:
        if n_cols < 3:
            raise ValueError(f"{parsed.path}: expected at least 3 columns for x,y,z; found {n_cols}.")
        return ColumnModel(x_idx=0, y_idx=1, z_idx=2, vector_groups=[])

    normed = [_normalize_name(c) for c in parsed.columns]

    def pick_axis(axis: str) -> Optional[int]:
        axis = axis.lower()
        synonyms = {
            "x": {"x", "coordx", "xcoord", "posx", "xpos", "positionx", "xposition"},
            "y": {"y", "coordy", "ycoord", "posy", "ypos", "positiony", "yposition"},
            "z": {"z", "coordz", "zcoord", "posz", "zpos", "positionz", "zposition"},
        }[axis]

        for idx, n in enumerate(normed):
            if n in synonyms:
                return idx

        for idx, n in enumerate(normed):
            if n == axis:
                return idx
            if n.startswith(axis) and len(n) <= 4:
                return idx
            if n.endswith(axis) and (n.startswith("coord") or n.startswith("pos") or n.startswith("position")):
                return idx

        return None

    x_idx = pick_axis("x")
    y_idx = pick_axis("y")
    z_idx = pick_axis("z")

    if x_idx is None or y_idx is None or z_idx is None:
        raise ValueError(
            f"{parsed.path}: could not detect x/y/z from header.\n"
            f"Header columns: {parsed.columns}\n"
            "Tip: Use recognizable names like x y z, coordX coordY coordZ; "
            "or remove the header to use the first 3 numeric columns."
        )

    coord_set = {x_idx, y_idx, z_idx}

    base_to_axes: Dict[str, Dict[str, int]] = {}
    for idx, n in enumerate(normed):
        if idx in coord_set:
            continue
        if not n:
            continue
        last = n[-1]
        if last not in ("x", "y", "z"):
            continue
        base = n[:-1]
        if not base:
            continue
        base_to_axes.setdefault(base, {})[last] = idx

    vector_groups: List[Tuple[int, int, int]] = []
    for axes_map in base_to_axes.values():
        if all(k in axes_map for k in ("x", "y", "z")):
            vector_groups.append((axes_map["x"], axes_map["y"], axes_map["z"]))

    return ColumnModel(x_idx=x_idx, y_idx=y_idx, z_idx=z_idx, vector_groups=vector_groups)


def _parse_axes(s: str) -> List[str]:
    s = s.strip().lower()
    if not s:
        raise ValueError("Empty axis selection.")
    parts = re.split(r"[^xyz]+", s)
    joined = "".join(parts)
    axes = sorted(set(joined))
    if not axes or any(a not in ("x", "y", "z") for a in axes):
        raise ValueError(f"Invalid axis selection: {s!r}")
    return axes


def generate_reflections(axes: Sequence[str]) -> List[Tuple[int, int, int]]:
    axes = [a.lower() for a in axes]
    combos = []
    for r in range(0, len(axes) + 1):
        for subset in itertools.combinations(axes, r):
            sx, sy, sz = 1, 1, 1
            if "x" in subset:
                sx = -1
            if "y" in subset:
                sy = -1
            if "z" in subset:
                sz = -1
            combos.append((sx, sy, sz))
    combos = sorted(combos, key=lambda t: (t != (1, 1, 1), t))
    return combos


def transform_vectors(
    data: np.ndarray,
    model: ColumnModel,
    sign_xyz: Tuple[int, int, int],
    vector_groups: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    sx, sy, sz = sign_xyz
    out = data.copy()
    out[:, model.x_idx] *= sx
    out[:, model.y_idx] *= sy
    out[:, model.z_idx] *= sz
    for vx, vy, vz in vector_groups:
        out[:, vx] *= sx
        out[:, vy] *= sy
        out[:, vz] *= sz
    return out


def _dedupe_by_xyz(data: np.ndarray, model: ColumnModel, tol: float) -> np.ndarray:
    if tol <= 0:
        return data
    xyz = data[:, [model.x_idx, model.y_idx, model.z_idx]]
    keys = np.round(xyz / tol).astype(np.int64)
    seen: Dict[Tuple[int, int, int], int] = {}
    keep: List[int] = []
    for i, k in enumerate(keys):
        kt = (int(k[0]), int(k[1]), int(k[2]))
        if kt in seen:
            continue
        seen[kt] = i
        keep.append(i)
    return data[np.asarray(keep, dtype=np.int64)]


def _format_row(row: np.ndarray, delimiter: str) -> str:
    return delimiter.join(f"{v:.15e}" for v in row)


def write_file(out_path: Path, parsed: ParsedFile, out_data: np.ndarray, out_delimiter: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if parsed.preamble_lines:
        lines.extend(parsed.preamble_lines)
    if parsed.header_line is not None:
        lines.append(parsed.header_line)
    for row in out_data:
        lines.append(_format_row(row, out_delimiter))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(msg).strip()
    v = input(f"{msg} [{default}]: ").strip()
    return v if v else default


def _gather_inputs_from_path(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    raise ValueError(f"Input path does not exist or is not a file/folder: {path}")


def _choose_no_header_vector_mode_interactive() -> str:
    print("\nNo header detected in at least one file.")
    print("How should columns AFTER x,y,z be treated?")
    print("  1) All scalars (no sign flips)  [safest default]")
    print("  2) Grouped triples as vectors: (c4,c5,c6) is vector1 (x,y,z), (c7,c8,c9) vector2, ...")
    while True:
        choice = _prompt("Select 1 or 2", "1")
        if choice in ("1", "2"):
            return "scalars" if choice == "1" else "triples"
        print("Invalid selection. Please enter 1 or 2.")


def _vector_groups_for_no_header(n_cols: int, mode: str) -> List[Tuple[int, int, int]]:
    if mode == "scalars":
        return []
    if mode == "triples":
        extra = n_cols - 3
        if extra <= 0:
            return []
        if extra % 3 != 0:
            print(
                f"Warning: remaining columns ({extra}) are not divisible by 3; "
                "treating all as scalars for this file."
            )
            return []
        groups = []
        start = 3
        for i in range(extra // 3):
            groups.append((start + 3 * i + 0, start + 3 * i + 1, start + 3 * i + 2))
        return groups
    raise ValueError(f"Unknown no-header vector mode: {mode}")


def process_one_file(
    in_path: Path,
    axes: Sequence[str],
    out_dir: Path,
    pattern: str,
    dedupe_tol: float,
    no_header_vector_mode: str,
) -> Optional[Path]:
    try:
        parsed = parse_file(in_path)
    except Exception as e:
        print(f"[ERROR] Failed parsing {in_path}: {e}", file=sys.stderr)
        return None

    try:
        model = detect_columns(parsed)
    except Exception as e:
        print(f"[ERROR] Column detection failed for {in_path}: {e}", file=sys.stderr)
        return None

    if parsed.columns is None:
        vector_groups = _vector_groups_for_no_header(parsed.data.shape[1], no_header_vector_mode)
    else:
        vector_groups = model.vector_groups

    reflections = generate_reflections(axes)
    pieces = [transform_vectors(parsed.data, model, s, vector_groups) for s in reflections]
    out_data = np.vstack(pieces)

    expected = parsed.data.shape[0] * len(reflections)
    before = out_data.shape[0]

    out_data = _dedupe_by_xyz(out_data, model, tol=dedupe_tol)

    after = out_data.shape[0]
    if dedupe_tol > 0 and after != before:
        print(
            f"[INFO] {in_path.name}: dedupe removed {before - after} row(s) "
            f"(expected={expected}, wrote={after}, tol={dedupe_tol:g})"
        )
    else:
        print(f"[INFO] {in_path.name}: expected={expected}, wrote={after}")

    out_delim = parsed.delimiter if parsed.delimiter in (",", "\t") else "\t"

    stem = in_path.stem
    ext = in_path.suffix
    out_name = pattern.format(stem=stem, ext=ext)
    if not out_name.endswith(ext):
        out_name = out_name + ext
    out_path = out_dir / out_name

    try:
        write_file(out_path, parsed, out_data, out_delim)
    except Exception as e:
        print(f"[ERROR] Failed writing {out_path}: {e}", file=sys.stderr)
        return None

    return out_path


def _demo() -> None:
    n = 50
    theta = np.linspace(-np.pi / 2, np.pi / 2, n)
    r = 1.0
    x = np.abs(r * np.cos(theta))
    y = r * np.sin(theta)
    z = np.zeros_like(theta)
    ux, uy, uz = x.copy(), y.copy(), z.copy()
    data = np.column_stack([x, y, z, ux, uy, uz]).astype(np.float64)

    parsed = ParsedFile(
        path=Path("<demo>"),
        preamble_lines=["# Demo dataset: x y z ux uy uz"],
        header_line="x y z ux uy uz",
        delimiter=None,
        columns=["x", "y", "z", "ux", "uy", "uz"],
        data=data,
    )
    model = detect_columns(parsed)

    axes = ["x", "y"]
    reflections = generate_reflections(axes)
    pieces = [transform_vectors(parsed.data, model, s, model.vector_groups) for s in reflections]
    out_data = np.vstack(pieces)

    print("DEMO")
    print(f"  input rows:  {parsed.data.shape[0]}")
    print(f"  axes:        {''.join(axes).upper()}")
    print(f"  reflections: {len(reflections)}")
    print(f"  expected:    {parsed.data.shape[0] * len(reflections)}")
    print(f"  wrote:       {out_data.shape[0]}  (dedupe disabled by default)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mirror partial lens datasets to full lens via symmetry reflections.")
    ap.add_argument("--demo", action="store_true", help="Run built-in demo and exit.")
    ap.add_argument("--input", type=str, default=None, help="Input file path or folder path. If omitted, prompts.")
    ap.add_argument("--axes", type=str, default=None, help="Axes to mirror across: X, Y, Z, XY, XZ, YZ, XYZ.")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory. If omitted, prompts.")
    ap.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Output filename pattern using {stem} and {ext}. Default: {stem}_full{ext}",
    )
    ap.add_argument(
        "--dedupe-tol",
        type=float,
        default=0.0,
        help="Deduplicate by (x,y,z) after mirroring using this tolerance. Default 0 disables dedupe.",
    )
    ap.add_argument(
        "--no-header-vectors",
        choices=("scalars", "triples", "ask"),
        default="ask",
        help="When a file has no header: treat extra cols as scalars, grouped triples as vectors, or ask.",
    )

    args = ap.parse_args(argv)

    if args.demo:
        _demo()
        return 0

    in_path_s = args.input or _prompt("Input path (file): ")
    in_path = Path(in_path_s).expanduser()

    while True:
        try:
            axes_s = args.axes or _prompt("Symmetry axes (X, Y, Z, XY, XZ, YZ, XYZ or comma-separated like x,y): ")
            axes = _parse_axes(axes_s)
            break
        except Exception as e:
            print(f"Invalid axes: {e}")

    out_dir_s = args.out_dir or _prompt("Output directory", str(in_path.parent / "symmetrized_out"))
    out_dir = Path(out_dir_s).expanduser()

    pattern = args.pattern or _prompt("Output filename pattern (use {stem} and {ext})", "{stem}_full{ext}")

    try:
        inputs = _gather_inputs_from_path(in_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if not inputs:
        print("[ERROR] No .txt files found to process.", file=sys.stderr)
        return 2

    no_header_vector_mode = args.no_header_vectors
    if no_header_vector_mode == "ask":
        any_no_header = False
        for p in inputs:
            try:
                parsed = parse_file(p)
                if parsed.columns is None:
                    any_no_header = True
                    break
            except Exception:
                continue
        no_header_vector_mode = _choose_no_header_vector_mode_interactive() if any_no_header else "scalars"

    print("\nProcessing:")
    print(f"  inputs:   {len(inputs)} file(s)")
    print(f"  axes:     {''.join(axes).upper()}")
    print(f"  out_dir:  {out_dir}")
    print(f"  pattern:  {pattern}")
    print(f"  dedupe:   tol={args.dedupe_tol:g} (0=disabled)")
    print(f"  no-header extra cols: {no_header_vector_mode}")

    ok = 0
    for p in inputs:
        out_path = process_one_file(
            in_path=p,
            axes=axes,
            out_dir=out_dir,
            pattern=pattern,
            dedupe_tol=args.dedupe_tol,
            no_header_vector_mode=no_header_vector_mode,
        )
        if out_path is None:
            continue
        ok += 1
        print(f"[OK] {p.name} -> {out_path}")

    print(f"\nDone. Wrote {ok}/{len(inputs)} file(s).")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
