# Deployment and Execution Guide

**Project:** MoleculeIQ  
**Target Environment:** Local Python Development / Academic Workstation  

---

## 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.6, Windows 11).
- Modern web browser (Chrome, Edge, Firefox, Safari).

## 2. Installation Steps

1. **Clone or Navigate to Repository:**
   ```bash
   git clone <repo-url>
   cd moleculeiq
   ```

2. **Create and Activate Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Tests:**
   ```bash
   python -m pytest tests/ -v
   ```

## 3. Running the Application

Launch the Streamlit web interface with:
```bash
streamlit run app.py
```

The application will launch and open in your default browser at:
`http://localhost:8501`

---

## 4. Retraining Predictive Models (Optional)

Model artifacts for ClinTox and all 12 Tox21 endpoints are already trained and bundled in `artifacts/`. If you wish to retrain from raw benchmark datasets:

```bash
python -c "
from src.models.clintox import train_clintox_model
from src.models.tox21 import train_tox21_all_endpoints

train_clintox_model()
train_tox21_all_endpoints()
"
```
