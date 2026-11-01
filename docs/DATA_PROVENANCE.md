# Data Provenance and Lineage

**Directive Version:** 5.0 (Phase 2E Open-Source Compliance Release)<br>
**PRD / SRD Source:** PRD v1.0 & SRD v1.0<br>

---

## 1. ClinTox Benchmark Dataset
- **Origin:** MoleculeNet Benchmark (Wu et al., 2018, *Chemical Science*, 9(2): 513–530).
- **Primary Source Lineage:**
  - FDA approval records sourced from the **SWEETLEAD database** (Novick et al., 2013, *PLoS ONE*, 8(11): e79252).
  - Clinical trial failure data due to human clinical toxicity sourced from the **AACT database** (Gayvert et al., 2016, *Cell Chemical Biology*, 23(9): 1194–1201).
- **Target Definition:** `CT_TOX` (1 = Clinical trial failure due to toxicity; 0 = FDA-approved / passed clinical development).
- **Structure Representation:** SMILES column parsed and canonicalized via RDKit `Chem.MolToSmiles(..., isomericSmiles=True)`.
- **Deduplication Policy:** Canonicalized and deduplicated to 1,461 unique chemical structures (resolving 19 stereochemical and duplicate conflicts with conservative-toxic prioritization).
- **Licensing:** Scientific benchmark compilation curated by MoleculeNet and distributed via DeepChem. Underlying SWEETLEAD data is subject to Creative Commons Attribution (CC BY 4.0); AACT clinical trial records are public domain. Academic and research redistribution is permitted with scientific citation of Wu et al. (2018) and Novick et al. (2013).

---

## 2. Tox21 Data Challenge Dataset
- **Origin:** US Toxicology in the 21st Century (Tox21) Consortium (NIH NCATS, EPA, NIEHS/NTP, FDA).
- **Assay Targets:** 12 specific in vitro high-throughput screening assays:
  - Nuclear Receptors: `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
  - Stress Response: `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`
- **Data Governance Policy:**
  - Independent modeling per endpoint.
  - Rows with null or empty labels for a given endpoint are excluded from that specific endpoint's dataset.
  - Missing assay outcomes are NEVER imputed as negative.
- **Legal Status:** Public Domain (U.S. Federal Government work under 17 U.S.C. § 105).

---

## 3. Approved Small Molecule Reference Collection (`data/reference/approved_drugs.csv`)
- **Compilation Structure:** A dual-provenance collection of 1,396 approved small-molecule drugs:
  1. **ChEMBL Curated Anchors (12 records):** Key reference compounds (Aspirin, Acetaminophen, Ibuprofen, Caffeine, Atorvastatin, Metformin, Omeprazole, Penicillin G, Amoxicillin, etc.) curated with explicit ChEMBL IDs (`CHEMBL25`, `CHEMBL112`, etc.) and tagged `source = ChEMBL_Curated_Anchor`.
     - **Licensing:** Governed by **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)**.
  2. **ClinTox FDA-Approved Subset (1,384 records):** Approved small molecules extracted from the ClinTox benchmark (`FDA_APPROVED == 1`) and tagged `source = FDA_Approved_ClinTox`.
     - **Licensing:** Governed by MoleculeNet / SWEETLEAD attribution terms (CC BY 4.0).
- **Derived Properties:** Pre-computed 2048-bit Morgan circular fingerprints (radius=2) and RDKit physicochemical descriptors (`formula`, `molecular_weight`, `logp`, `lipinski_pass`).
- **Data Asset License:** In accordance with CC BY-SA 3.0, `data/reference/approved_drugs.csv` is distributed as a database asset under CC BY-SA 3.0. MoleculeIQ application source code remains under the MIT License.

---

## 4. PubMed Literature Stream & Evaluation Benchmark
- **Live Search Origin:** National Center for Biotechnology Information (NCBI) PubMed database via E-utilities (`esearch`, `efetch`).
- **API Governance:** Strictly throttled to $\le 3\text{ requests/second}$ for unauthenticated access in compliance with NCBI terms of service. Secure TLS certificate validation leverages operating system native trust stores.
- **Benchmark Evaluation Architecture (Phase 2E):**
  - **Manifest File:** `data/benchmarks/literature/pubmed_benchmark_manifest.json` contains query definitions, PMIDs, citation metadata (title, authors, year, URL), and character counts.
  - **Relevance Annotations (`data/benchmarks/literature/relevance_annotations_v2.json`):** Contains 198 sentence-level relevance grades (0–3), PMIDs, article sentence indices, evaluative rationales, and cryptographic verification hashes (SHA-256) authored for information retrieval evaluation. Fully excerpt-free: contains zero verbatim abstract sentence excerpts.
  - **Scholarly Benchmark & Copyright Notice:**
    - The relevance annotations are created strictly as scholarly benchmark material designed to evaluate and benchmark scientific information retrieval algorithms (MRR, nDCG@5, Precision@k).
    - Source publications are identified by PubMed Identifiers (PMIDs) and bibliographic metadata in the benchmark manifest.
    - Public distribution contains zero commercial publisher abstract text; sentence text is deterministically reconstructed in memory from the locally hydrated benchmark corpus via `reconstruct_annotated_sentences()` and verified against `sentence_hash_sha256`.
    - End-users and researchers utilizing this benchmark are responsible for complying with the original publishers' terms of use, applicable copyright statutes, and relevant research exemptions in their respective jurisdictions.
  - **Hydration Workflow:** Verbatim commercial journal abstract texts are not permanently bundled in the Git repository. Users hydrate the local benchmark file (`data/benchmarks/literature/pubmed_benchmark_v1.json`) on demand via:
    ```bash
    python scripts/hydrate_pubmed_benchmark.py
    ```
    This avoids bundling or redistributing third-party copyrighted journal text in the public Git repository while preserving reproducible automated benchmarking.

---

## 5. Component Provenance, Rights, and Licensing Boundary Matrix

To ensure absolute legal clarity for third-party adopters and compliance reviewers, MoleculeIQ formally partitions rights, licenses, and distribution channels across all codebase and data layers:

| Component Layer | Asset Description | Primary Source / Origin | Governing License / Legal Status | Distribution Method | Redistribution & Usage Terms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MoleculeIQ Application Source** | Python modules (`src/`), orchestration, contracts, UI components (`app.py`), tests | Original authorship (Amrita Dahiya & contributors) | **MIT License** | Git Repository | Unrestricted commercial and non-commercial use with copyright notice retention. |
| **ChEMBL Reference Anchors** | 12 curated drug records in `data/reference/approved_drugs.csv` (`ChEMBL_Curated_Anchor`) | ChEMBL database (EMBL-EBI, Wellcome Genome Campus) | **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)** | Git Repository (Data Asset) | Requires attribution and ShareAlike for derivative database distributions. Does not infect MIT Python code. |
| **ClinTox Benchmark Subset** | 1,384 FDA-approved records in `approved_drugs.csv` & training CSVs | MoleculeNet (Wu et al. 2018), SWEETLEAD, AACT | **CC BY 4.0 / Public Domain (Attribution)** | Git Repository | Permitted with citation of Wu et al. (2018) and Novick et al. (2013). |
| **Tox21 Challenge Data** | In vitro bioassay outcomes across 12 endpoints (`data/raw/tox21.csv.gz`) | NIH NCATS, EPA, NIEHS/NTP, FDA Tox21 Consortium | **U.S. Government Work / Public Domain (17 U.S.C. § 105)** | Git Repository | Unrestricted worldwide redistribution; does not imply government endorsement. |
| **PubMed Benchmark Manifest & Annotations** | Queries, PMIDs, citation metadata, and sentence-level relevance grades (`data/benchmarks/literature/`) | MoleculeIQ local curation & NCBI citation indices | **Scholarly Benchmark Material** | Git Repository | Scholarly benchmark for IR evaluation. Completely excerpt-free; sentence text reconstructed in-memory from local hydrated corpus. |
| **PubMed Full Abstracts** | Verbatim journal abstract texts (`pubmed_benchmark_v1.json`) | Commercial scientific publishers via NCBI E-utilities | **Publisher / Author Copyright** | Excluded from Git (`.gitignore`); hydrated locally on-demand | Hydrated locally by end-user via `scripts/hydrate_pubmed_benchmark.py`. Users must respect publisher terms and applicable law. |
| **Trained ML Models & Calibration Curves** | 32 release artifacts: 16 binary assets (14 `.joblib` models, 2 `.npz` fingerprint matrices) and 16 JSON metadata/benchmark files (`artifacts/clintox/`, `artifacts/tox21/`) | Fitted locally via scikit-learn on ClinTox and Tox21 | **Trained Statistical Artifacts** *(Derived from open datasets)* | GitHub Release Assets (`moleculeiq-models-v5.0.0.tar.gz`) | Verified via SHA-256 (`artifacts/checksums.sha256`). Subject to upstream dataset training disclosures. |
| **Pre-computed Morgan Fingerprint Matrices** | Numpy `.npz` arrays of 2048-bit Morgan circular fingerprints | Computed deterministically via RDKit from dataset SMILES | **Derived Computational Representations** | GitHub Release Assets | Representations of public domain and open chemical structures. |
| **Third-Party Libraries** | RDKit, scikit-learn, Streamlit, pandas, numpy, scipy, shap, anyio | PyPI ecosystem packages | **BSD 3-Clause, Apache 2.0, MIT** | `requirements.txt` | Standard open-source dependency compliance. |
| **3D Molecular Visualization** | 3Dmol.js JavaScript library | Nicholas Rego & David Koes, University of Pittsburgh | **BSD 3-Clause License** | Runtime CDN (`https://3dmol.org`) | Attributed in UI footer and visualization components. |

> **IMPORTANT NOTICE ON MATHEMATICAL & COMPUTATIONAL ARTIFACTS:**
> In accordance with scientific data governance standards, MoleculeIQ does not assert that machine-learning model weights, hyperplanes, Platt calibration curves, or pre-computed topological fingerprint arrays are automatically unrestricted public domain assets merely because they represent numerical or computational matrices. Rather, these artifacts are explicitly documented as trained statistical representations derived from the ClinTox and Tox21 datasets. While the governing inputs (MoleculeNet and NIH Tox21) permit open academic and research utilization, downstream commercial re-users must evaluate upstream provenance for their specific jurisdictions and regulatory regimes.
