# Marimo Notebooks — Bank Client Analysis

This folder contains interactive [Marimo](https://marimo.io) notebooks for bank client segmentation.

## Files

| File | Description |
|------|-------------|
| `bank_clients_analysis.py` | EDA + K-Medoids clustering notebook |
| `assets/` | Drop `.svg` files here — they are auto-discovered and rendered in the notebook |

## Running Locally

```bash
# From the repository root
cd BusinessCase1

# Install marimo (add to venv)
pip install marimo

# Run the notebook
marimo run marimo/bank_clients_analysis.py
# OR open in edit mode
marimo edit marimo/bank_clients_analysis.py
```

## Adding SVG Diagrams

Place any `.svg` files in `marimo/assets/`. The notebook scans the folder at startup and displays all SVGs in the **SVG Assets** panel (Part 0).

The image titles shown in the browser tab thumbnails (e.g. `pca_gower_pam_compatibility`, `weighted_gower_decision`) correspond to the filenames without extension — name your SVGs accordingly.

## Requirements

All dependencies are in `BusinessCase1/requirements.txt`. Additional packages needed for marimo:

```
marimo
```
