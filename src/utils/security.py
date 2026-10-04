"""Security safeguards: CSV formula injection defense, path traversal guard, and file validation."""

import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import pandas as pd

MAX_BATCH_ROWS = 5000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_csv_cell(val: Any) -> Any:
    """
    Sanitize a cell value to prevent CSV / Formula injection (CWE-1236).

    Prepend a single quote (') if the string begins with characters that
    spreadsheet software (Excel, LibreOffice, Google Sheets) interprets
    as executable formulas or DDE macro commands: '=', '+', '-', '@', '\\t', '\\r'.
    """
    if not isinstance(val, str):
        return val

    # If the string begins with a formula trigger, neutralize it with a leading quote
    if val.startswith(FORMULA_PREFIXES):
        return f"'{val}"

    return val


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with all string cells sanitized against formula injection.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if (
            pd.api.types.is_string_dtype(df_clean[col])
            or pd.api.types.is_object_dtype(df_clean[col])
            or str(df_clean[col].dtype) in ("str", "object", "string")
        ):
            df_clean[col] = df_clean[col].map(lambda x: sanitize_csv_cell(x) if isinstance(x, str) else x)
    return df_clean


def safe_join_path(base_dir: Union[str, Path], relative_path: str) -> Path:
    """
    Safely resolve a path relative to a base directory, guarding against path traversal attacks.

    Raises
    ------
    ValueError
        If the resolved path falls outside base_dir.
    """
    base = Path(base_dir).resolve()
    # Strip any leading slashes/backslashes to avoid absolute override
    clean_rel = relative_path.lstrip("/\\")
    resolved = (base / clean_rel).resolve()

    if not str(resolved).startswith(str(base)):
        raise ValueError(f"Path traversal detected: '{relative_path}' resolves outside '{base}'")

    return resolved


def validate_upload_content(file_bytes: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> Tuple[bool, Optional[str]]:
    """
    Validate binary content of uploaded files for size limits and null byte injection.
    """
    if len(file_bytes) > max_bytes:
        return False, f"File size ({len(file_bytes)} bytes) exceeds maximum limit of {max_bytes} bytes."

    if b"\x00" in file_bytes:
        return False, "File contains forbidden null bytes (potential binary/malicious payload)."

    return True, None
