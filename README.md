# construction-diff

Detect construction progress by comparing point cloud scans over time. Designed for the [Rohbau3D](https://github.com/Photogrammetry-Bonn/Rohbau3D) dataset format.

## Features

- Load Rohbau3D `.npy` scans (`coord.npy`, `color.npy`, `intensity.npy`, `normal.npy`)
- Global registration (FPFH + RANSAC) with ICP refinement using Open3D
- Diff computation via KD-tree nearest-neighbour distance thresholding
- Colour-coded visualisation: new (green), removed (red), unchanged (grey)

## Installation

```bash
pip install -e .
```

## Usage

### Align two scans

```bash
construction-diff align /path/to/scan_t0 /path/to/scan_t1 -o transform.npy
```

### Compute difference

```bash
construction-diff diff /path/to/scan_t0 /path/to/scan_t1 -t transform.npy -o diff.npz
```

### Generate report

```bash
construction-diff report /path/to/scan_t0 /path/to/scan_t1 -o report.png
```

## License

MIT
