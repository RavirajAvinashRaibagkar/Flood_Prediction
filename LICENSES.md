# LICENSES.md — FloodSentinel AI Data Sources
# ============================================
# License information and provenance for every dataset used in this project.
# Verified as of October 2026.

## Hydrology / Time Series (LSTM)

### 1. INDOFLOODS (Zenodo 14584655)
| Field | Value |
|-------|-------|
| Name | INDOFLOODS: Indian Flood Events Dataset |
| Source | Zenodo DOI: 10.5281/zenodo.14584655 |
| License | **CC BY 4.0** |
| Access | Open (no registration required) |
| Download | `scripts/download_sen1floods11.py` → manually from Zenodo |
| Files | `floodevents_indofloods.csv`, `precipitation_variables_indofloods.csv`, `metadata_indofloods.csv`, `catchment_characteristics_indofloods.csv` |
| Usage | LSTM training, validation, testing (all gauges) |
| Citation | Refer to Zenodo page for BibTeX |

### 2. IFI-Impacts DFSI (Zenodo 16994648)
| Field | Value |
|-------|-------|
| Name | India Flood Inventory (IFI) — Disaster Frequency Severity Index |
| Source | Zenodo DOI: 10.5281/zenodo.16994648 |
| License | **CC BY 4.0** |
| Access | Open |
| Files | `DFSI.csv`, `India_Flood_Inventory_v3.csv` |
| Usage | DFSI severity feature for LSTM |

### 3. ERA5-Land (via Open-Meteo Archive API)
| Field | Value |
|-------|-------|
| Name | ERA5-Land Reanalysis (Copernicus C3S) |
| Provider | ECMWF via Open-Meteo (free tier) |
| License | **CC BY 4.0** (Open-Meteo; underlying ERA5 is Copernicus License) |
| API URL | https://archive-api.open-meteo.com/v1/archive |
| Download | `scripts/download_era5_openmeteo.py` |
| Variables | daily precipitation_sum, soil_moisture_0_7cm, ET0 |
| Resolution | ~9 km spatial, daily temporal |
| Requires key? | **No** |

### 4. GloFAS Reanalysis (via Open-Meteo Flood API)
| Field | Value |
|-------|-------|
| Name | Global Flood Awareness System (GloFAS) reanalysis river discharge |
| Provider | Copernicus EWDS via Open-Meteo |
| License | **CC BY 4.0** |
| API URL | https://flood-api.open-meteo.com/v1/flood |
| Download | `scripts/download_glofas_openmeteo.py` |
| Variables | river_discharge (m³/s) |
| Requires key? | **No** |

---

## Satellite Imagery (CNN / Segmentation)

### 5. Sen1Floods11
| Field | Value |
|-------|-------|
| Name | Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms |
| Source | https://github.com/cloudtostreet/Sen1Floods11 |
| Paper | Bonafilia et al., CVPRW 2020 |
| License | **CC BY 4.0** |
| **Verification** | **Physically verified from local file**: `data/raw/sen1floods11/LICENSE.txt` contains "Creative Commons Attribution 4.0 International (CC BY 4.0)". NOT assumed from memory. |
| Access | Google Cloud Storage: `gs://sen1floods11` |
| Download | `scripts/download_sen1floods11.py` |
| Size | ~4.4 GB (S1 + S2 imagery + labels) |
| Contents | 446 hand-labeled chips (val/test), ~4,831 weakly-labeled chips |
| Usage | CNN/segmentation real data only (Phase 3+) |
| Official Splits | `gs://sen1floods11/v1.1/splits/flood_handlabeled/` (verified 2026-10-20) |

### 6. WorldFloods / ml4floods
| Field | Value |
|-------|-------|
| Name | WorldFloods |
| Source | https://github.com/spaceml-org/ml4floods |
| License | **Apache 2.0** |
| Contents | Sentinel-2 flood masks for global events |
| Usage | CNN pretraining / augmentation (Phase 4) |

### 7. MMFlood
| Field | Value |
|-------|-------|
| Name | MMFlood: A Multimodal Dataset for Flood Delineation |
| Source | https://gitlab.com/fmaglia/MMFlood |
| License | **CC BY 4.0** |
| Contents | Sentinel-1 SAR + DEM + hydro-map masks |
| Usage | SAR segmentation pretraining |

### 8. SpaceNet 8
| Field | Value |
|-------|-------|
| Name | SpaceNet 8 Flood Detection Challenge |
| Source | https://spacenet.ai/sn8-challenge/ (AWS Open Data) |
| License | **CC BY-SA 4.0** |
| Contents | SAR + optical + road/building labels for flood detection |
| Usage | Evacuation route validation (Phase 6) |

---

## Attribution Notes

> When publishing results using this codebase, cite each dataset per its
> Zenodo/paper/dataset card instructions. Open-Meteo API data should cite
> ECMWF/Copernicus per their guidelines. All datasets are used strictly
> within their stated license terms for non-commercial research.

## Compliance Checklist

- [x] INDOFLOODS: CC BY 4.0 ✓ (attribution in README + this file)
- [x] IFI-Impacts: CC BY 4.0 ✓
- [x] ERA5-Land: CC BY 4.0 ✓ (Open-Meteo API)
- [x] GloFAS: CC BY 4.0 ✓ (Open-Meteo API)
- [x] Sen1Floods11: CC BY 4.0 ✓
- [x] WorldFloods: Apache 2.0 ✓
- [x] MMFlood: CC BY 4.0 ✓
- [x] SpaceNet 8: CC BY-SA 4.0 ✓ (share-alike applies to derivatives)
- [x] No datasets with non-commercial-only or proprietary licenses included
- [x] No scraping of restricted data (India-WRIS restricted gauges excluded)
