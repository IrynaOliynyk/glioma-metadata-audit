# glioma-metadata-audit
Computational audit of WHO ICTRP glioma clinical trial metadata. This Python-based pipeline identifies structural degradation and semantic inconsistencies in registry data to evaluate its readiness for AI-driven drug discovery. Includes phase normalization and translational mapping.

# Clinical registry metadata as a hidden bottleneck in AI-driven drug discovery: a computational audit of translational phase data in glioma research

### Iryna Oliynyk (Lviv Medical University; Institute of Animal Biology NAAS)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18430433.svg)](https://doi.org/10.5281/zenodo.18430433)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Abstract
Clinical translation in glioma remains inefficient despite advances in computational drug discovery. This study analyzes 2,357 records from the WHO ICTRP registry using a deterministic Python-based validation pipeline. We identify a "technical vacuum" in phase reporting (579 missing entries) and severe terminological fragmentation in study-type fields. These findings demonstrate that semantic validation is a prerequisite for reliable AI-driven translational assessment.

## Data Access
The raw dataset retrieved on 03.01.2026 is permanently archived on Zenodo:
**[https://doi.org/10.5281/zenodo.18430433](https://doi.org/10.5281/zenodo.18430433)**

## Code Functionality
The provided Python script performs:
1. **Metadata Ingestion:** Loads ICTRP export while preserving semantic markers (N/A, NaN).
2. **Phase Normalization:** A regex-based engine that classifies over-specified narrative entries into standard FDA phases (0-IV).
3. **Translational Mapping:** Categorizes trials into four quadrants (Interventional, Observational, Patient Registries, Expanded Access).
4. **Statistical Auditing:** Calculates the *Reporting Gap (RG)* and *Maturity Ratio (MR)*.

## How to Run
1. Download `glioma_who_3.01.2026.xlsx` from the Zenodo link above (https://zenodo.org/records/18430433).
2. Ensure you have the required libraries installed:
   `pip install pandas matplotlib seaborn openpyxl`
3. Run the analysis script:
   `python_audit_pipeline.py`

## Citation
If you use this code or dataset, please cite:
Oliynyk, I. (2026). Dataset for: Clinical registry metadata as a hidden bottleneck in AI-driven drug discovery. Zenodo. https://doi.org/10.5281/zenodo.18430433
