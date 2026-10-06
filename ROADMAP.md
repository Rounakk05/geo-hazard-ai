# ROADMAP — Geospatial AI Framework for Multi-Hazard Disaster Detection and Damage Assessment

> **How to use this file with Claude in VS Code:** put it in the repo root as `ROADMAP.md`, then tell Claude:
> *"Read ROADMAP.md. Execute Phase 0 first. After each phase, stop, summarise what you built, and wait for me to say 'next'."*
> Work phase by phase. Do not ask Claude to build everything in one go.

---

## 0. Project context (Claude: read this first)

**Team:** Anmol Badoni, Nikunj Kaushik, Rounak Khurana (UPES Dehradun, B.Tech CSE AIML). Mentor: Mr. Aryan.
**Duration:** 12 weeks. **Compute:** assume student resources only (Kaggle GPU / Google Colab / one consumer GPU). No cluster.

**Goal:** One modular pipeline that takes multi-source satellite imagery (Sentinel-1 SAR, Sentinel-2 optical, DEM/slope, pre/post-event pairs) and produces:
1. Hazard masks (flood, landslide, wildfire/burn scar, + earthquake/cyclone via building damage)
2. Change detection between pre- and post-event images
3. Building-level damage classes: **undamaged / partially damaged / severely damaged**
4. Severity maps, affected area (km²), confidence scores
5. GIS-ready outputs: **GeoTIFF + GeoJSON**, shown in a **dashboard**

### Scope decision (important, read this)
Five hazards with five different datasets is the biggest risk in this project. Build it in **tiers**, so there is always a working demo:

| Tier | Scope | Must finish by |
|------|-------|----------------|
| **Tier 1 (core, mandatory)** | Flood segmentation (Sentinel-1/2) + building damage (xBD, pre/post Siamese) + GIS export + dashboard | Week 10 |
| **Tier 2 (should have)** | Landslide + wildfire/burn-scar segmentation using the same modular training code | Week 11 |
| **Tier 3 (stretch)** | Earthquake and cyclone shown through building-damage inference on xBD/BRIGHT events; YOLO building detector; foundation-model backbone comparison | Week 12 / if time |

The synopsis lists all five hazards. Earthquake and cyclone are handled as **damage-assessment events** (building damage on event imagery), not as separate hazard-segmentation models. State this clearly in the report and viva.

---

## 1. Architecture

```
Data sources (S1, S2, DEM, xBD/BRIGHT, hazard datasets)
        │
[1] Ingestion & preprocessing   → co-register, resample to common grid, cloud mask, normalise, tile
        │
[2] Feature engineering         → NDWI/MNDWI, NDVI, NBR/dNBR, VV/VH, VV-VH ratio, slope, elevation
        │
[3] Hazard identification       → U-Net style segmentation (one head/config per hazard)
        │
[4] Temporal change detection   → Siamese encoder (shared weights) on pre/post
        │
[5] Damage assessment           → Siamese/ConvNeXt classifier or segmentation head (3 classes)
        │
[6] GIS conversion              → raster → polygons, area km², confidence, severity attrs → GeoTIFF/GeoJSON
        │
[7] Dashboard & decision support→ Streamlit + folium/leafmap map, filters, stats, download
```

**Design rule:** every hazard is a *config file*, not a code fork. A new hazard or sensor should only need a new dataset class + a YAML config.

---

## 2. Repository structure (Claude: create this in Phase 0)

```
geo-hazard-ai/
├── ROADMAP.md
├── CLAUDE.md                 # short project rules for Claude (see Section 9)
├── README.md
├── pyproject.toml / requirements.txt
├── Makefile                  # make setup | data | train | eval | demo
├── configs/
│   ├── flood.yaml
│   ├── landslide.yaml
│   ├── wildfire.yaml
│   └── damage_xbd.yaml
├── data/                     # gitignored
│   ├── raw/  interim/  processed/  splits/
├── src/geohazard/
│   ├── data/                 # downloaders, datasets, transforms, tiling, splits
│   ├── features/             # spectral indices, SAR features, terrain
│   ├── models/               # unet, siamese, convnext_siamese, fusion, heads
│   ├── training/             # train loop, losses, metrics, callbacks
│   ├── inference/            # sliding-window predict, tta, stitching
│   ├── gis/                  # polygonize, area stats, geojson/geotiff export
│   ├── dashboard/            # streamlit app
│   └── utils/                # logging, seed, io, crs helpers
├── notebooks/                # EDA, visual checks, results
├── scripts/                  # CLI entry points
├── tests/
├── outputs/                  # predictions, figures, tables (gitignored except samples)
└── docs/                     # report figures, architecture diagram
```

---

## 3. Environment & tools

**Python 3.10–3.11**, one virtualenv/conda env.

| Purpose | Packages |
|---------|----------|
| Deep learning | `torch`, `torchvision`, `timm`, `segmentation-models-pytorch`, `pytorch-lightning`, `torchmetrics`, `albumentations` |
| Geo / raster | `rasterio`, `geopandas`, `shapely`, `pyproj`, `fiona`, `rioxarray`, `xarray`, `rasterstats` |
| Satellite access | `pystac-client`, `planetary-computer`, `stackstac` or `odc-stac`, `earthengine-api`, `sentinelsat` / Copernicus Data Space API |
| Remote-sensing DL | `torchgeo` (datasets, pretrained weights, samplers) |
| Detection (Tier 3) | `ultralytics` (YOLO) |
| ML utils | `scikit-learn`, `numpy`, `pandas`, `opencv-python`, `scikit-image` |
| Tracking | `wandb` or `mlflow` (pick one), `tqdm` |
| Dashboard | `streamlit`, `folium`, `streamlit-folium` (or `leafmap`), `plotly` |
| Quality | `pytest`, `ruff`, `black`, `pre-commit` |
| Desktop GIS (manual checking) | QGIS (free) — open every exported GeoTIFF/GeoJSON in it to verify |

**Accounts you will need (free):** Google Earth Engine, Copernicus Data Space Ecosystem, Microsoft Planetary Computer (open STAC, no key for most data), xView2 challenge site (for xBD), Kaggle, Hugging Face, Weights & Biases.

---

## 4. Datasets (verify license + download link yourself before use)

I have listed well-known datasets by name. **Check each one's official page for current download instructions and license**, because links and hosting change. Use datasets only under their license terms and cite them in the report.

### 4.1 Core (Tier 1)

| Dataset | Used for | Sensor / format | Notes |
|---------|----------|-----------------|-------|
| **Sen1Floods11** | Flood segmentation | Sentinel-1 (VV/VH) + Sentinel-2 chips, 512×512, 11 flood events, hand-labelled + weakly-labelled | Best primary flood dataset for this project. Hosted on Google Cloud Storage; use `gsutil`. |
| **xBD (xView2)** | Building damage, pre/post change detection | ~0.5 m Maxar optical pre/post pairs, building polygons, 4 damage levels, 19 disaster events (floods, hurricanes/cyclones, earthquake, wildfire, volcano, tsunami) | Requires free registration on the xView2 site. **Not Sentinel**; it is high-resolution optical. Non-commercial research license. Map labels to project classes (see Section 6.3). |
| **Copernicus DEM GLO-30 / SRTM** | Elevation + slope | GeoTIFF | Pull through Planetary Computer, Earth Engine or OpenTopography. Derive slope with `numpy`/`rasterio`/`richdem`. |

### 4.2 Extension hazards (Tier 2)

| Dataset | Hazard | Notes |
|---------|--------|-------|
| **Landslide4Sense** (competition dataset) | Landslides | Sentinel-2 bands + DEM-derived slope/elevation, 128×128 patches with masks. Directly matches "optical + slope" in the synopsis. |
| **CaBuAr** (California Burned Areas) | Wildfire burn scars | Sentinel-2 pre/post-fire with burn masks (Hugging Face). Good for bi-temporal change detection. |
| **Sen2Fire** | Wildfire | Sentinel-2 + Sentinel-5P patches with fire masks. Alternative to CaBuAr. |
| **WorldFloods (ML4Floods)** | Flood (second dataset) | Sentinel-2 flood masks across many events; use for **cross-dataset generalisation** tests. |
| **OMBRIA** or **SEN12-FLOOD** | Flood (SAR-focused / time series) | Optional extras for SAR-optical fusion experiments. |

### 4.3 Stretch / multi-hazard damage (Tier 3)

| Dataset | Use |
|---------|-----|
| **BRIGHT** (optical + SAR building damage) | Multimodal building-damage across several event types incl. earthquakes; supports cross-event generalisation. |
| **DisasterM3** | Cited in your synopsis (26,988 bi-temporal images, 36 events). Check its repository/Hugging Face page for access; use for evaluation or qualitative demos only if practical. |
| **RescueNet** (Hurricane Michael UAV) / **FloodNet** (Hurricane Harvey UAV) | UAV-level segmentation; matches the UAV direction in your literature review. |
| **IBTrACS** + **ERA5** | Cyclone track and rainfall/wind context layers for the cyclone case study (not for training). |
| **Copernicus Emergency Management Service (CEMS) Rapid Mapping** | Real event polygons for validating flood/earthquake/wildfire maps and case studies. |
| **NASA FIRMS / EFFIS** | Active fire / burned-area context for wildfire case studies. |

### 4.4 Pretrained weights to try (Tier 3, optional)
`torchgeo` pretrained ResNet/ViT (SSL4EO-S12, SeCo), IBM-NASA **Prithvi-EO**, **Clay**, SatMAE, DINOv2 / SAM for ConvNeXt/ViT comparisons.
Always start with an **ImageNet-pretrained ConvNeXt-Tiny / ResNet-34** baseline via `timm`; it is enough for the project.

### 4.5 Where to get raw satellite data for your own case study
- **Microsoft Planetary Computer** (STAC): Sentinel-1 RTC, Sentinel-2 L2A, Copernicus DEM
- **Google Earth Engine**: fast cloud-side compositing and export
- **Copernicus Data Space Ecosystem**: official Sentinel archive
- **ASF Vertex**: Sentinel-1 SAR download

**Case study suggestion:** pick 1–2 real events (e.g. a recent Indian flood such as Assam/Bihar/Kerala, a Uttarakhand landslide, a wildfire) and run the final pipeline end-to-end on new Sentinel data. This makes the demo far more convincing.

---

## 5. Phase-by-phase plan (matches the synopsis timeline)

Each phase lists: **Goal → Tasks → Deliverables → Done when**. Claude: complete one phase at a time.

---

### PHASE 0 — Setup (before Week 1 ends)
**Goal:** reproducible repo + environment.

Tasks
1. Create the repo structure from Section 2.
2. Create `requirements.txt`, `Makefile`, `.gitignore` (ignore `data/`, `outputs/`, checkpoints, `.env`).
3. Add `CLAUDE.md` (Section 9) and `configs/*.yaml` stubs.
4. Add `utils/seed.py`, `utils/logging.py`, config loader (YAML → dataclass/pydantic).
5. Add a smoke test: import all modules, load a config, create a tiny random tensor through a U-Net.
6. Set up GitHub repo, branches (`main`, `dev`), and pre-commit (ruff + black).

Deliverables: runnable skeleton, `make setup` works, `pytest` passes.
**Done when:** a fresh clone → `make setup && pytest` succeeds.

---

### PHASE 1 — Literature, requirements, datasets (Weeks 1–2)
**Goal:** finalise scope and secure the data.

Tasks
1. Each team member reads and summarises 2–3 of the 7 references in the synopsis (1 paragraph each + "what we take from it").
2. Write `docs/requirements.md`: inputs, outputs, classes, metrics, non-goals.
3. Register on xView2, Earth Engine, Copernicus, Kaggle, W&B.
4. Download **Sen1Floods11** and **xBD** first. Then Landslide4Sense and CaBuAr if time allows.
5. Write `scripts/download_*.py` for every dataset (idempotent, checks file hashes/counts, logs sizes).
6. Write `notebooks/01_eda_*.ipynb` per dataset: class balance, image sizes, bands, CRS, example visualisations.
7. Decide label mapping and class weights (Section 6.3).

Deliverables: data in `data/raw/`, EDA notebooks, `docs/requirements.md`, dataset table with license + size.
**Done when:** every Tier-1 dataset loads in a notebook and you can plot an image with its mask.

---

### PHASE 2 — Preprocessing & feature pipeline (Weeks 3–5)
**Goal:** a clean, tested data pipeline that matches Stage 1 and Stage 2 of the methodology.

Tasks
1. **Alignment:** verify CRS, resample/co-register to a common grid (`rasterio.warp.reproject`).
2. **Cleaning:** cloud/shadow masking (S2 SCL band or QA60), nodata handling, SAR speckle filtering (Lee/median) or log-scaling, dB conversion for Sentinel-1.
3. **Normalisation:** per-band mean/std computed on the **training split only**; store in JSON.
4. **Tiling:** 256×256 (or 512 → random crop 256) with overlap for inference; keep georeference per tile.
5. **Feature engineering** in `src/geohazard/features/`:
   - Optical: NDWI, MNDWI, NDVI, NBR (and dNBR for fire), BSI
   - SAR: VV, VH, VV/VH ratio, VV−VH, pre/post difference
   - Terrain: elevation, slope, (optional) aspect, HAND/TWI for floods
6. **Splits:** by **event/region**, not random tiles, to avoid geographic leakage (the synopsis requires this). Save split lists in `data/splits/*.json`.
7. **Datasets & loaders:** PyTorch `Dataset` classes returning `{image, mask, meta}`; augmentations via albumentations (flips, rot90, light brightness/contrast for optical only, no colour jitter on SAR).
8. **Tests:** unit tests for indices (known values), tiling (stitch → original), and split leakage (no event in two splits).

Deliverables: `processed/` tiles, `splits/`, dataloaders, tests.
**Done when:** `python scripts/check_pipeline.py --config configs/flood.yaml` prints shapes, band stats, and saves a sample grid image; leakage test passes.

---

### PHASE 3 — Baseline & multimodal models (Weeks 6–8)
**Goal:** trained models for Tier 1, then extend.

#### 3A. Flood segmentation (Week 6)
1. Baseline: `smp.Unet` (ResNet-34/ConvNeXt-Tiny encoder), input = S2 only.
2. Variant 2: S1 only (VV, VH).
3. Variant 3: **early fusion** (S1 + S2 + DEM/slope channels).
4. Variant 4: **late/attention fusion** (two encoders → feature concat or cross-attention → decoder).
5. Loss: BCE + Dice (or Focal + Dice). Optimiser AdamW, cosine LR, mixed precision, early stopping.
6. Log IoU, F1, precision, recall; save best checkpoint by val IoU.

#### 3B. Building damage (Week 6–7)
1. Task A: building localisation (U-Net on pre-image), or use xBD building polygons directly as input for the damage stage in the first version.
2. Task B: damage classification with a **Siamese network**: shared encoder on pre and post, feature difference/concatenation, decoder or classification head. Compare:
   - Siamese ResNet-34 U-Net
   - Siamese ConvNeXt-Tiny
   - Attention-based fusion of pre/post features (multimodal attention reference in your literature review)
3. Loss: weighted cross-entropy or focal loss (damage classes are heavily imbalanced).
4. Class mapping and metrics in Section 6.3.

#### 3C. Tier 2 hazards (Week 7–8)
1. Reuse the same training script with `configs/landslide.yaml` and `configs/wildfire.yaml`.
2. Landslide: Sentinel-2 bands + slope + elevation → U-Net.
3. Wildfire: pre/post Sentinel-2 + dNBR → Siamese or early-fusion U-Net.

#### 3D. Hazard identification module (Week 8)
- Small classifier or rule layer that routes input to the right hazard model (use metadata/user selection first; learned classifier is optional). Output a hazard label + confidence.

Deliverables: checkpoints, training logs (W&B), a results table per model, `notebooks/03_results.ipynb`.
**Done when:** Tier 1 models beat a trivial baseline and the numbers are reproducible with fixed seeds.

---

### PHASE 4 — Evaluation, ablation, GIS conversion (Weeks 9–10)
**Goal:** prove it works and turn pixels into GIS.

Evaluation tasks
1. Metrics: accuracy, precision, recall, F1, IoU (segmentation); per-class F1 and macro-F1, confusion matrix (damage). For xBD optionally report the xView2-style combined score.
2. **Ablations (required for a strong report):**
   - S2 only vs S1 only vs S1+S2 vs S1+S2+DEM
   - Single-image vs bi-temporal (change detection)
   - Early fusion vs attention fusion
   - ResNet vs ConvNeXt encoder
   - With/without class weighting
3. **Cross-region generalisation:** train on some events, test on held-out events/regions; also test flood model trained on Sen1Floods11 on WorldFloods or a CEMS event.
4. **Robustness:** simulate clouds (mask optical) and check that SAR keeps performance; add noise tests.
5. Error analysis: save the 20 worst and 20 best predictions with commentary.

GIS conversion tasks (`src/geohazard/gis/`)
1. Sliding-window inference with overlap + blending → full-scene probability raster.
2. Threshold → binary mask → morphological cleanup → `rasterio.features.shapes` → polygons.
3. Compute attributes per polygon: area (km², use an equal-area/UTM CRS), mean confidence, severity class, hazard type, event date, source scenes.
4. Building-level polygons: join damage class to building footprints (`geopandas.sjoin`/zonal majority vote).
5. Exports: **GeoTIFF** (mask, probability, severity map) and **GeoJSON** (polygons) in EPSG:4326, plus a summary CSV (area by class).
6. Verify every output in QGIS (correct CRS, alignment with basemap).

Deliverables: evaluation tables + figures, ablation chart, exported GIS layers for 2–3 scenes.
**Done when:** you can run `python scripts/predict.py --scene <id> --hazard flood` and open the outputs in QGIS with sensible alignment.

---

### PHASE 5 — Dashboard, documentation, final presentation (Weeks 11–12)
**Goal:** a demo-ready system and final report.

Dashboard (Streamlit)
1. Sidebar: choose hazard, event/scene, pre/post slider, confidence threshold, damage-class filter.
2. Map: basemap + hazard mask overlay + damage polygons coloured by class (folium/leafmap); popups with coordinates, confidence, area.
3. Panels: affected area (km²), counts per damage class, bar/pie charts, confusion matrix and metrics page.
4. Buttons: download GeoJSON/GeoTIFF/CSV.
5. "Upload your own pre/post GeoTIFF" mode (optional but impressive).
6. Run on cached example outputs so the demo works offline.

Documentation & submission
1. README with install, data download, train, evaluate, demo commands.
2. Architecture diagram, pipeline figure, results tables, sample maps.
3. Final report: Introduction → Literature/SWOT → Objectives → Methodology → Experiments → Results → Limitations → Future work → References.
4. Slides + a 2–3 minute screen-recorded demo as backup.
5. Tag a release (`v1.0`), freeze requirements, store final checkpoints (HF Hub or Drive link).

**Done when:** a teammate can clone the repo, follow the README, and reproduce the demo.

---

## 6. Technical details Claude should follow

### 6.1 Model configs (starting points)
- Input size 256×256, batch 8–16 (reduce if OOM), AdamW lr 1e-4 (encoder) / 1e-3 (decoder), weight decay 1e-4, 30–60 epochs, cosine schedule, AMP on.
- Seeds fixed (42) and logged; save config with every checkpoint.
- Use `torch.utils.data.DataLoader(num_workers>0)`; cache normalised tiles as `.npy`/`.tif` for speed.

### 6.2 Fusion options to implement (for the "compare architectures" objective)
1. Early fusion (stack channels)
2. Two-stream encoders + feature concatenation
3. Cross-attention / channel-attention fusion block
4. Siamese shared-weight encoder with |f_pre − f_post| and concat(f_pre, f_post)

### 6.3 Damage class mapping (xBD → project classes)
| xBD label | Project class |
|-----------|---------------|
| no-damage | 0 — Undamaged |
| minor-damage + major-damage | 1 — Partially damaged |
| destroyed | 2 — Severely damaged |

Use class-weighted loss or oversampling; report **per-class F1** and macro-F1, not just accuracy.

### 6.4 Metrics cheat sheet
- Segmentation: IoU, Dice/F1, precision, recall
- Classification: macro-F1, per-class F1, confusion matrix
- Detection (Tier 3 YOLO): mAP@0.5
- GIS check: polygon count, total area, comparison to reference area where available

### 6.5 Coding standards
- Type hints, docstrings, small functions, config-driven, no hard-coded paths.
- Every module has a test or a smoke script.
- Never commit data, API keys or checkpoints; use `.env`.

---

## 7. Compute plan (student-friendly)
- Develop and debug locally on small subsets (`--debug` flag: 50 tiles, 2 epochs).
- Train on **Kaggle** (weekly GPU quota) or **Colab**; sync code with Git, data via Kaggle datasets/Drive.
- Use AMP, 256 crops, gradient accumulation, and `timm` small encoders to fit memory.
- Checkpoint every epoch so interrupted sessions can resume.

---

## 8. Team split (suggested; adjust as you like)
| Member | Main ownership | Support |
|--------|----------------|---------|
| Member A | Data pipeline: downloads, preprocessing, features, splits | Evaluation |
| Member B | Models: flood/landslide/wildfire segmentation, fusion variants | Dashboard |
| Member C | Damage assessment (xBD Siamese), GIS conversion, dashboard | Docs |
| All | Literature review, report, slides, case study | |

Weekly: 30-min sync, update a `PROGRESS.md` log, short demo to mentor Mr. Aryan every 2 weeks.

---

## 9. Suggested `CLAUDE.md` (create this file in Phase 0)

```markdown
# Project rules
- Project: Geospatial AI multi-hazard detection + damage assessment (see ROADMAP.md).
- Work one ROADMAP phase at a time. Stop at the end of each phase and summarise.
- Python 3.10+, PyTorch, rasterio, geopandas. Config-driven (YAML in configs/).
- Split data by event/region, never random tiles. Compute normalisation stats on train only.
- Never commit data, checkpoints, or secrets. Keep paths in configs/.env.
- Add tests for every new module. Run `pytest` and `ruff` before finishing a task.
- Prefer small, readable functions. Log metrics to W&B/MLflow.
- If a dataset URL, API or license is unclear, ask me instead of guessing.
```

---

## 10. Prompts you can paste into Claude (one per phase)

**Phase 0:** "Read ROADMAP.md and CLAUDE.md. Execute Phase 0: create the repo structure, requirements, Makefile, config loader, and smoke tests. Show me the file tree when done."

**Phase 1:** "Execute Phase 1 for Sen1Floods11 and xBD only. Write download scripts and EDA notebooks. Tell me the manual steps I must do (registrations, tokens)."

**Phase 2:** "Execute Phase 2. Implement preprocessing, feature engineering, event-based splits, datasets/dataloaders and tests for the flood and xBD configs."

**Phase 3A/B:** "Implement the flood U-Net (S2-only, S1-only, early fusion, attention fusion) and the xBD Siamese damage model. Add a train script driven by configs and a `--debug` mode."

**Phase 3C:** "Add landslide and wildfire configs/datasets using the existing training code."

**Phase 4:** "Implement evaluation, ablation runner, cross-region test, and the GIS module (sliding-window inference, polygonisation, area stats, GeoTIFF/GeoJSON export)."

**Phase 5:** "Build the Streamlit dashboard per Phase 5, plus README and architecture diagram."

---

## 11. Risks and fallbacks

| Risk | Mitigation |
|------|------------|
| Five hazards too ambitious | Follow the tier plan; Tier 1 alone is a complete project |
| Dataset access delays (xBD registration) | Register in week 1; start with Sen1Floods11 meanwhile |
| Class imbalance in damage labels | Weighted/focal loss, oversampling, macro-F1 reporting |
| GPU limits | Smaller encoders, 256 crops, AMP, Kaggle/Colab, subset training |
| Geographic leakage | Event-level splits + automated leakage test |
| Cloud-covered optical data | SAR branch + fusion ablation (this *is* a result for your report) |
| xBD is not Sentinel resolution | Be explicit: Sentinel for hazard extent, xBD high-res for building damage |
| Dashboard not ready | Build a minimal map page by Week 9 and improve it |

---

## 12. Final deliverables checklist
- [ ] GitHub repo with README, tests, configs, reproducible commands
- [ ] Trained checkpoints (flood, damage; plus landslide/wildfire if done)
- [ ] Results tables, ablations, confusion matrices, example maps
- [ ] Exported GeoTIFF/GeoJSON samples that open correctly in QGIS
- [ ] Streamlit dashboard + demo video
- [ ] Final report and presentation (with limitations and SWOT revisited)
- [ ] Reference list matching the synopsis (7 core references + datasets used)

---
