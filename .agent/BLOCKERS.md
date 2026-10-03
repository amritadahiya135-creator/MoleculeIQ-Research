# Blockers and Open Issues

## Active Blockers
*None currently.*

## Tracked Risks & Dependencies
1. **PyTorch / sentence-transformers dependency:**
   - *Impact:* Module M09 (PubMed semantic sentence retrieval with `all-MiniLM-L6-v2`) requires sentence embedding generation.
   - *Status:* `sentence_transformers` and `torch` are not installed in the global Python environment.
   - *Plan:* In Task 06, attempt `pip install sentence-transformers torch` or provide an integrated semantic search fallback (e.g. TF-IDF / BM25 or lightweight onnx / pure python embedding adapter) if full PyTorch wheel installation is heavy or restricted.

2. **ChEMBL / PubMed Network Rate Limits:**
   - *Impact:* Live REST requests during evaluation or demo could experience latency or network failure.
   - *Mitigation:* Bundle a local curated reference cache of approved/investigational drugs with precomputed fingerprints and identifiers for guaranteed offline execution.
