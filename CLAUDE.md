# Project rules

**Project:** Geospatial AI Framework for Multi-Hazard Disaster Detection and Damage Assessment (UPES major project). Full plan is in `ROADMAP.md`.

## How to work
- Work on **one ROADMAP phase at a time**. At the end of each phase, stop, summarise what you built, list what I must do manually (registrations, downloads, tokens), and wait for me to say "next".
- Follow the tier plan in ROADMAP.md: Tier 1 (flood + xBD building damage + GIS export + dashboard) comes first. Do not start Tier 2/3 until Tier 1 works.
- If a dataset URL, API, or license is unclear, **ask me instead of guessing**. If the docs conflict with ROADMAP.md, prefer the docs, tell me, and propose an update to the roadmap.

## Tech stack
- Python 3.10+, PyTorch, timm, segmentation-models-pytorch, rasterio, geopandas, shapely, Streamlit.
- Config-driven: hazards, datasets and model settings live in `configs/*.yaml`. Adding a hazard should need a new dataset class and a new YAML, not a code fork.

## Data rules
- Split data by **event/region**, never by random tiles. Add and keep a test that checks no event appears in two splits.
- Compute normalisation statistics on the **training split only**.
- Keep georeference (CRS, transform) with every tile so predictions can be exported as GeoTIFF/GeoJSON.
- Use an equal-area or UTM CRS when computing areas (km²); export final layers in EPSG:4326.
- Damage classes: 0 = undamaged, 1 = partially damaged (xBD minor + major), 2 = severely damaged (xBD destroyed).

## Code standards
- Type hints, docstrings, small readable functions, no hard-coded paths (use configs or `.env`).
- Fix seeds (42) and log them. Save the config alongside every checkpoint.
- Add a test or smoke script for every new module. Run `pytest` and `ruff` before declaring a task done.
- Provide a `--debug` flag on training scripts (tiny subset, 1–2 epochs) so code can be checked quickly on CPU.
- Log metrics to W&B or MLflow (pick one and stay consistent).

## Never
- Never commit datasets, model checkpoints, large rasters, API keys, or `.env` files.
- Never report accuracy alone for imbalanced damage classes; always include per-class F1 and macro-F1.
- Never invent results, metrics, or citations. If something was not run, say so.

## Evaluation and outputs
- Segmentation: IoU, F1/Dice, precision, recall. Damage: macro-F1, per-class F1, confusion matrix.
- Every prediction run should be able to export: hazard mask GeoTIFF, probability GeoTIFF, severity GeoTIFF, polygon GeoJSON (with area_km2, confidence, severity, hazard), and a summary CSV.
- Verify exported layers open correctly in QGIS.
