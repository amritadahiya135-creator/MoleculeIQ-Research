# Security and Resilience Specification

**Project:** MoleculeIQ  
**Directive Version:** 4.0  

---

## 1. Input Sanitization & Ingestion Security
1. **SMILES Input Security:**
   - User-supplied chemical line notations are strictly parsed using RDKit's C++ parser inside controlled try/except blocks.
   - Malformed, oversized, or unparseable inputs are rejected before downstream processing.
   - Code execution via chemical notation injection is fundamentally prevented.
2. **File Upload Security (CSV):**
   - Uploaded files are handled strictly as structured tabular data using `pd.read_csv()`.
   - File extensions are restricted to `.csv`.
   - Executable code inside CSV cells is never evaluated or executed.
   - Row-level errors are isolated without crashing the application process.

## 2. Model Deserialization Security
1. **Local Artifact Boundaries:**
   - Machine learning models serialized via `joblib` are loaded strictly from the designated local application directories (`artifacts/clintox` and `artifacts/tox21`).
   - The application rejects user-uploaded or arbitrary external pickled models to eliminate Python deserialization vulnerabilities (`RCE`).

## 3. Network & External Request Controls
1. **NCBI E-utilities & ChEMBL Requests:**
   - Bounded timeouts (default 15 seconds) on all HTTP requests to prevent hanging threads.
   - Rate-limiting adherence to prevent denial-of-service throttling from public NCBI services.
   - Queries are cached in memory to eliminate redundant external requests.
   - Offline fallback: Curated approved drug reference cache is bundled locally, guaranteeing uninterrupted chemical similarity search even during total network disconnection.

## 4. Privacy & Data Retention
1. **No User Accounts or Profile Tracking:**
   - The platform does not collect user credentials or track user browsing behavior.
   - Compound screening jobs are executed in local memory and are not stored permanently or transmitted to third parties.
