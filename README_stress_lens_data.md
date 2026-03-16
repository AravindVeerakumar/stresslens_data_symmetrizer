# Stress Lens Data Symmetrizer

Python utility to convert partial 3D stress datasets into full symmetric datasets by mirroring points across selected coordinate planes.

## What this project does

Finite element and multiphysics simulations are often exported only for part of a lens geometry - for example a half-model or quadrant - to reduce compute cost. This script reconstructs the full dataset by applying geometric symmetry to the available points.

Given an input `.txt` file with at least `x`, `y`, and `z` columns, the script:

- reads the point dataset
- detects coordinate columns automatically when a header exists
- mirrors data across one or more symmetry planes
- preserves scalar fields such as stress values
- flips vector components correctly when mirrored
- writes the expanded dataset to a new output file

This repository currently includes:

- **Input dataset**: `Input/semi processed 3d stress_MPa.txt`
- **Generated output dataset**: `output/complete stress data Mpa.txt`
- **Main script**: `Symmetrize_stressdata.py`
- **Reference documentation**: PDF and DOCX notes used during implementation

## Example from this repository

The included sample demonstrates a partial stress dataset being expanded into a full dataset.

- input rows: **2379**
- output rows: **4758**
- symmetry expansion: **2x**

That output size is consistent with applying symmetry once with deduplication disabled.

## How symmetry works

The script supports reflection across the three coordinate planes:

- **X symmetry**: reflect across the YZ plane  
  `(x, y, z) -> (-x, y, z)`
- **Y symmetry**: reflect across the XZ plane  
  `(x, y, z) -> (x, -y, z)`
- **Z symmetry**: reflect across the XY plane  
  `(x, y, z) -> (x, y, -z)`

If you select multiple axes, the script generates all symmetry combinations.

Examples:

- `X` -> 2 copies total
- `XY` -> 4 copies total
- `XYZ` -> 8 copies total

## Scalar vs vector handling

The script treats physical quantities correctly during reflection.

### Scalar fields

Scalar values remain unchanged during mirroring.

Examples:

- normal stress components stored as scalar columns
- temperature
- von Mises stress
- pressure

### Vector fields

If the file contains vector components such as `Ux`, `Uy`, `Uz`, the mirrored coordinate direction changes sign.

Examples:

- mirror in `X` -> `Ux` sign flips
- mirror in `Y` -> `Uy` sign flips
- mirror in `Z` -> `Uz` sign flips

If the file has no header, the script can either:

- treat all columns after `x`, `y`, `z` as scalars
- treat the remaining columns as grouped vector triples

## Input format

The script accepts text datasets with:

- optional comments and blank lines
- optional header row
- comma, tab, or space-separated columns
- at least 3 numeric columns for coordinates

### Headered files

If a header exists, the script tries to detect coordinate columns using names like:

- `x`, `y`, `z`
- `coordX`, `coordY`, `coordZ`
- `posX`, `posY`, `posZ`

### Headerless files

If no header exists, the script assumes:

- column 1 = `x`
- column 2 = `y`
- column 3 = `z`

## Output behavior

The script preserves as much structure as possible:

- keeps initial comments and blank lines
- keeps the original header if present
- keeps comma or tab delimiters when detected
- otherwise writes tab-delimited output
- writes numeric values in scientific notation with high precision

By default, **deduplication is disabled**. That means the row count is intentionally multiplied exactly by the number of reflection combinations.

If needed, you can enable deduplication with a small tolerance to remove repeated points lying on the symmetry plane.

## Project structure

```text
stress-lens-data-symmetrizer/
├── Input/
│   └── semi processed 3d stress_MPa.txt
├── output/
│   └── complete stress data Mpa.txt
├── Symmetrize_stressdata.py
├── python script implementation for creating symmetry stress data.pdf
└── README.md
```

## Requirements

- Python 3.9+
- NumPy

Install dependency:

```bash
pip install numpy
```

## How to run

### Interactive mode

Run the script:

```bash
python Symmetrize_stressdata.py
```

You will be prompted for:

1. **Input path** - file or folder to process
2. **Symmetry axes** - `X`, `Y`, `Z`, `XY`, `XZ`, `YZ`, or `XYZ`
3. **Output directory**
4. **Output filename pattern**
5. **How to treat non-coordinate columns** for headerless files

### Command-line mode

Example:

```bash
python Symmetrize_stressdata.py \
  --input "Input/semi processed 3d stress_MPa.txt" \
  --axes Z \
  --out-dir output \
  --pattern "complete stress data Mpa"
```

For explicit scalar handling on headerless data:

```bash
python Symmetrize_stressdata.py \
  --input "Input/semi processed 3d stress_MPa.txt" \
  --axes Z \
  --out-dir output \
  --pattern "complete stress data Mpa" \
  --no-header-vectors scalars
```

## Useful options

```bash
python Symmetrize_stressdata.py --help
```

Available options include:

- `--input` - input file or folder
- `--axes` - symmetry axes to apply
- `--out-dir` - output folder
- `--pattern` - output naming pattern using `{stem}` and `{ext}`
- `--dedupe-tol` - optional deduplication tolerance
- `--no-header-vectors` - `scalars`, `triples`, or `ask`
- `--demo` - run built-in demonstration

## Notes and limitations

- The script assumes symmetry-based reconstruction is physically valid for the exported model.
- Without a header, the script cannot infer field semantics beyond the first three coordinate columns.
- Deduplication is disabled by default to preserve exact reflection-based row multiplication.
- This repository is focused on text-based stress data expansion, not mesh reconstruction or visualization.

## Typical use case

This project is useful when you have simulation data exported from only part of a symmetric lens geometry and need a full dataset for:

- post-processing
- visualization
- downstream analysis
- ML or statistical workflows
- full-domain data archiving

## Author

Aravind Veerakumar

## License

Add a license if you plan to share or reuse this code publicly.
