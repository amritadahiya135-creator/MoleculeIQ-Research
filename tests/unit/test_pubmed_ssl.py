import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from src.literature.pubmed_client import PubMedClient


class TestPubMedSSLSecurity:
    """Verifies that PubMedClient strictly enforces SSL certificate verification."""

    def test_ssl_strict_mode_by_default(self, monkeypatch):
        """Verify that SSLError is raised by default and does not fallback silently."""
        monkeypatch.delenv("MOLECULEIQ_ALLOW_INSECURE_SSL", raising=False)

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.SSLError("Self-signed certificate untrusted")

        client = PubMedClient(session=mock_session)

        # In strict mode, an SSLError must not be bypassed with verify=False
        with pytest.raises(requests.exceptions.SSLError):
            client._get("https://eutils.ncbi.nlm.nih.gov/test", {})

    def test_ssl_insecure_fallback_only_when_explicitly_configured(self, monkeypatch):
        """Verify that fallback to verify=False only occurs with explicit environment opt-in."""
        monkeypatch.setenv("MOLECULEIQ_ALLOW_INSECURE_SSL", "1")

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.SSLError("SSL Failure")

        client = PubMedClient(session=mock_session)

        with patch("requests.get") as mock_fallback_get:
            mock_fallback_resp = MagicMock()
            mock_fallback_resp.status_code = 200
            mock_fallback_get.return_value = mock_fallback_resp

            resp = client._get("https://eutils.ncbi.nlm.nih.gov/test", {})

            # Must have called fallback with verify=False
            mock_fallback_get.assert_called_once_with(
                "https://eutils.ncbi.nlm.nih.gov/test",
                params={},
                verify=False,
                timeout=15,
            )
            assert resp.status_code == 200
