# Data Provenance and Lineage

**Directive Version:** 4.0  
**PRD / SRD Source:** PRD v1.0 & SRD v1.0  

---

## 1. ClinTox Benchmark Dataset
- **Origin:** MoleculeNet Benchmark / Broad Institute / NIH
- **Primary Source:** FDA approval records and NIH clinical trial failure databases.
- **Target Definition:** `CT_TOX` (1 = Clinical trial failure due to toxicity; 0 = FDA-approved / passed clinical development).
- **Structure Representation:** SMILES column parsed and canonicalized via RDKit `Chem.MolToSmiles(..., isomericSmiles=True)`.
- **Sample Distribution:** Approximately 1,478 compounds (severely imbalanced, ~112 toxic vs. 1,366 non-toxic/approved).
- **Splitting Strategy:** Stratified random split (80% train, 20% test) with fixed random seed = 42.

## 2. Tox21 Data Challenge Dataset
- **Origin:** US Toxicology in the 21st Century Consortium (NIH NCATS, EPA, NTP, FDA).
- **Assay Targets:** 12 specific in vitro high-throughput screening assays:
  - Nuclear Receptors: `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
  - Stress Response: `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`
- **Data Governance Policy:**
  - Independent modeling per endpoint.
  - Rows with null or empty labels for a given endpoint are excluded from that specific endpoint's dataset.
  - Missing assay outcomes are NEVER imputed as negative.

## 3. Approved Small Molecule Reference Collection (ChEMBL-Derived)
- **Origin:** ChEMBL Database (EMBL-EBI) approved drug collection (Maximum Clinical Phase = 4).
- **Curation Criteria:**
  - Small organic molecules (MW between 100 and 900 Da).
  - Valid RDKit parseable SMILES.
  - Salt stripping and counterion removal applied where appropriate.
  - Pre-computed 2048-bit Morgan circular fingerprints (radius=2).
  - Bundled locally in `data/reference/approved_drugs.csv` to ensure high-performance offline similarity search without REST API downtime risks.

## 4. PubMed Literature Stream
- **Origin:** National Center for Biotechnology Information (NCBI) PubMed database via E-utilities (`esearch`, `efetch`).
- **Data Extracted:** PubMed ID (PMID), Title, Abstract text, Publication Year, Authors, Journal.
- **Processing:** Segmented into individual sentences with preserved parent article provenance.
