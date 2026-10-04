"""Unit tests for application security safeguards (CWE-1236, path traversal, batch limits)."""

import pytest
import pandas as pd
from pathlib import Path

from src.utils.security import (
    sanitize_csv_cell,
    sanitize_dataframe_for_csv,
    safe_join_path,
    validate_upload_content,
    MAX_BATCH_ROWS,
)
from src.services.orchestration import ScreeningOrchestrator


class TestSecurityDefenses:
    """Tests for formula injection defense, path traversal guard, and file validation."""

    def test_sanitize_csv_cell_formula_prefixes(self):
        # Malicious formula payloads
        assert sanitize_csv_cell("=1+1") == "'=1+1"
        assert sanitize_csv_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
        assert sanitize_csv_cell("-SUM(A1:A10)") == "'-SUM(A1:A10)"
        assert sanitize_csv_cell("@HYPERLINK('http://evil.com')") == "'@HYPERLINK('http://evil.com')"
        assert sanitize_csv_cell("\tTabPrefix") == "'\tTabPrefix"
        assert sanitize_csv_cell("\rReturnPrefix") == "'\rReturnPrefix"

        # Safe inputs
        assert sanitize_csv_cell("Aspirin") == "Aspirin"
        assert sanitize_csv_cell("CC(=O)Oc1ccccc1C(=O)O") == "CC(=O)Oc1ccccc1C(=O)O"
        assert sanitize_csv_cell(123) == 123
        assert sanitize_csv_cell(45.67) == 45.67
        assert sanitize_csv_cell(None) is None

    def test_sanitize_dataframe_for_csv(self):
        df = pd.DataFrame({
            "Name": ["=MALICIOUS()", "Aspirin", "@SUM"],
            "MW": [180.15, 200.0, 300.5],
            "Score": [0.85, 0.92, 0.40],
        })
        clean_df = sanitize_dataframe_for_csv(df)

        assert clean_df["Name"].tolist() == ["'=MALICIOUS()", "Aspirin", "'@SUM"]
        # Numeric columns untouched
        assert clean_df["MW"].tolist() == [180.15, 200.0, 300.5]

    def test_safe_join_path(self, tmp_path):
        base_dir = tmp_path / "safe_dir"
        base_dir.mkdir()

        # Valid subpath
        valid = safe_join_path(base_dir, "sub/file.txt")
        assert valid == (base_dir / "sub" / "file.txt").resolve()

        # Malicious traversal
        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join_path(base_dir, "../../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            safe_join_path(base_dir, "..\\..\\windows\\system32")

    def test_validate_upload_content(self):
        # Valid content
        ok, err = validate_upload_content(b"smiles,name\nCCO,Ethanol\n")
        assert ok is True
        assert err is None

        # Null byte injection
        ok, err = validate_upload_content(b"smiles,name\x00malicious")
        assert ok is False
        assert "null bytes" in err.lower()

        # Oversized content
        ok, err = validate_upload_content(b"A" * 100, max_bytes=50)
        assert ok is False
        assert "exceeds maximum limit" in err.lower()

    def test_screen_batch_size_guardrail(self):
        orchestrator = ScreeningOrchestrator()
        giant_batch = ["CCO"] * (MAX_BATCH_ROWS + 1)
        with pytest.raises(ValueError, match="exceeds maximum supported limit"):
            orchestrator.screen_batch(giant_batch)
