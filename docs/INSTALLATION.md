# MoleculeIQ Installation & Deployment Guide

**Version:** 5.0 (Research Prototype Release Candidate)  
**Target Environment:** Local Python, Virtual Environments, and Docker Containers  

---

## 1. System Requirements

* **Operating System:** Linux (Ubuntu 22.04+ recommended), macOS (12+), or Windows 10/11 (64-bit).
* **Python Runtime:** Python 3.11 or Python 3.12.
* **Hardware:**
  * CPU: Minimum 2 cores (4+ cores recommended for parallel model inference).
  * RAM: Minimum 4 GB (8 GB recommended for in-memory ChEMBL reference search).
  * Disk: ~1.5 GB for repository, dependencies, and trained model artifacts.

---

## 2. Local Installation (Native Python)

### Step 1: Clone Repository
```bash
git clone https://github.com/moleculeiq/moleculeiq.git
cd moleculeiq
```

### Step 2: Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Linux Users:** RDKit drawing utilities require X11/cairo rendering libraries:
> ```bash
> sudo apt-get update && sudo apt-get install -y libxrender1 libxext6 libsm6
> ```

### Step 4: Verify Artifact Cryptographic Integrity
To guarantee that model weights, calibration records, and reference caches have not experienced data corruption or bit rot, run the SHA-256 integrity verification:
```bash
python -c "from src.utils.integrity import verify_artifact_checksums; res = verify_artifact_checksums(); print(res); assert res.is_valid"
```

### Step 5: Run Verification Test Suite
```bash
python -m pytest tests/unit/ tests/integration/ tests/system/ -v
```

### Step 6: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will launch on `http://localhost:8501`.

---

## 3. Containerized Deployment (Docker)

A hardened, multi-stage production Dockerfile is included at the repository root. It executes under a dedicated non-root application user (`moleculeiq`) with health check instrumentation.

### Build Docker Image
```bash
docker build -t moleculeiq:latest .
```

### Run Container
```bash
docker run -d \
  --name moleculeiq-app \
  -p 8501:8501 \
  --restart unless-stopped \
  moleculeiq:latest
```

### Verify Container Health
```bash
docker ps --filter "name=moleculeiq-app"
curl -f http://localhost:8501/_stcore/health
```

---

## 4. Environment Variables

The application operates with zero required environment variables (all paths are automatically anchored to `PROJECT_ROOT`). The following variables may be set for custom configurations:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MOLECULEIQ_DATA_DIR` | `<root>/data` | Root directory for raw and reference datasets |
| `MOLECULEIQ_ARTIFACTS_DIR` | `<root>/artifacts` | Directory holding trained models and calibration data |
| `REQUESTS_CA_BUNDLE` | System Default | Path to custom enterprise CA certificate bundle for PubMed queries |
| `STREAMLIT_SERVER_PORT` | `8501` | Network port for web UI binding |

---

## 5. Troubleshooting & Diagnostics

1. **RDKit Import Error (`DLL load failed` on Windows):**
   * Ensure Microsoft Visual C++ 2015–2022 Redistributable (x64) is installed.
2. **PubMed API Connection Timeout:**
   * MoleculeIQ includes an automatic fallback and caching mechanism. If operating behind an enterprise TLS proxy, set `REQUESTS_CA_BUNDLE` to your corporate certificate.
3. **Model Checksum Mismatch:**
   * If `verify_artifact_checksums()` reports a failure, re-download the affected model file or inspect `artifacts/checksums.sha256`.
