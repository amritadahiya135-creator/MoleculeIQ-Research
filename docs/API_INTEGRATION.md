# External API Integration Specification

**Project:** MoleculeIQ  
**Integrated External Services:** NCBI E-utilities (PubMed) & ChEMBL  

---

## 1. NCBI Entrez E-utilities (PubMed)

### Overview
NCBI Entrez E-utilities provides programmatic access to PubMed bibliographic data and abstracts. MoleculeIQ connects to NCBI to retrieve peer-reviewed scientific literature matching target molecular queries.

### Endpoints Used
1. **ESearch (`esearch.fcgi`):**
   - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
   - Parameters:
     - `db=pubmed`
     - `term={clean_query}`
     - `retmode=json`
     - `retmax={max_results}` (default: 5)
   - Function: Converts text queries into a list of relevant PubMed IDs (PMIDs).

2. **EFetch (`efetch.fcgi`):**
   - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi`
   - Parameters:
     - `db=pubmed`
     - `id={pmid_list}`
     - `retmode=xml`
   - Function: Retrieves full XML records containing Title, Abstract, Authors, Journal, and Publication Year.

### Rate Limiting & Resilience
- NCBI limits unauthenticated requests to 3 requests per second.
- Requests are executed synchronously with 15-second timeouts.
- Query results are cached in an in-memory dictionary to eliminate duplicate network calls.
- If network connection fails or NCBI is unreachable, the system produces a controlled status (`NO_ARTICLES_FOUND` or error message) rather than terminating the application.

---

## 2. ChEMBL Reference Drug Collection

### Architecture
To ensure high-throughput execution and eliminate runtime dependency on external ChEMBL REST server availability, approved small molecules are pre-curated from ChEMBL clinical phase 4 / FDA approval registries and bundled locally in `data/reference/approved_drugs.csv`.

### Reference Data Schema
- `reference_id`: Unique ChEMBL or FDA identifier (e.g. `CHEMBL25`).
- `name`: Common international nonproprietary drug name (e.g. Aspirin).
- `canonical_smiles`: Sanitized isomeric SMILES structure.
- `molecular_weight`: Physicochemical molecular weight in Daltons.
- `logp`: Calculated octanol-water partition coefficient.
- `lipinski_pass`: Boolean flag indicating Lipinski Rule of 5 compliance.
