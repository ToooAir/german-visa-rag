"""Unit tests for src/ingestion/cli.py"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from src.ingestion.cli import (
    _print_ingestion_summary,
    app,
    load_domain_configs,
    load_seed_urls,
)

runner = CliRunner()

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _seed_yaml(tmp_path: Path, extra_urls=None, domains=None, documents=None) -> Path:
    data = {}
    if extra_urls is not None:
        data["extra_urls"] = extra_urls
    if documents is not None:
        data["documents"] = documents
    if domains is not None:
        data["domains"] = domains
    p = tmp_path / "seed_urls.yml"
    p.write_text(yaml.dump(data))
    return p


_SUCCESS_RESULT = {
    "success": True,
    "run_id": "run-001",
    "documents_processed": 2,
    "chunks_ingested": 10,
    "chunks_skipped": 1,
    "errors": [],
    "documents_skipped_quota": 0,
}

_FAILURE_RESULT = {
    "success": False,
    "run_id": "run-002",
    "documents_processed": 0,
    "chunks_ingested": 0,
    "chunks_skipped": 0,
    "errors": [{"url": "https://a.com", "error": "timeout"}],
    "documents_skipped_quota": 0,
}

_QUOTA_RESULT = {
    "success": False,
    "quota_exhausted": True,
    "wait_seconds": 3600,
    "run_id": "run-003",
    "documents_processed": 0,
    "chunks_ingested": 0,
    "chunks_skipped": 0,
    "errors": [],
    "documents_skipped_quota": 1,
}


# ─── load_seed_urls ────────────────────────────────────────────────────────────


class TestLoadSeedUrls:
    def test_loads_extra_urls(self, tmp_path):
        p = _seed_yaml(tmp_path, extra_urls=[{"url": "https://a.com"}])
        result = load_seed_urls(str(p))
        assert result == [{"url": "https://a.com"}]

    def test_falls_back_to_documents_key(self, tmp_path):
        p = _seed_yaml(tmp_path, documents=[{"url": "https://b.com"}])
        result = load_seed_urls(str(p))
        assert result == [{"url": "https://b.com"}]

    def test_returns_empty_on_file_not_found(self):
        result = load_seed_urls("/nonexistent/path.yml")
        assert result == []

    def test_returns_empty_on_missing_key(self, tmp_path):
        p = tmp_path / "empty.yml"
        p.write_text(yaml.dump({"other_key": []}))
        result = load_seed_urls(str(p))
        assert result == []


# ─── load_domain_configs ──────────────────────────────────────────────────────


class TestLoadDomainConfigs:
    def test_loads_domains(self, tmp_path):
        p = _seed_yaml(tmp_path, domains=[{"domain": "example.com"}])
        result = load_domain_configs(str(p))
        assert result == [{"domain": "example.com"}]

    def test_returns_empty_on_error(self):
        result = load_domain_configs("/no/such/file.yml")
        assert result == []

    def test_returns_empty_when_no_domains_key(self, tmp_path):
        p = tmp_path / "nodomain.yml"
        p.write_text(yaml.dump({"extra_urls": []}))
        result = load_domain_configs(str(p))
        assert result == []


# ─── _print_ingestion_summary ────────────────────────────────────────────────


class TestPrintIngestionSummary:
    def test_prints_success_summary(self):
        from unittest.mock import patch as mpatch

        captured = []
        with mpatch("typer.echo", side_effect=lambda msg="": captured.append(str(msg))):
            _print_ingestion_summary(_SUCCESS_RESULT)
        output = "\n".join(captured)
        assert "2" in output  # documents_processed
        assert "10" in output  # chunks_ingested

    def test_prints_error_summary_and_writes_file(self, tmp_path):
        result = dict(_FAILURE_RESULT)
        with (
            patch("typer.echo"),
            patch("src.ingestion.cli.Path") as mock_path_cls,
        ):
            mock_path = MagicMock()
            mock_path.__truediv__ = MagicMock(return_value=mock_path)
            mock_path.parent = MagicMock()
            mock_path_cls.return_value = mock_path
            mock_open = MagicMock()
            mock_open.__enter__ = MagicMock(return_value=MagicMock())
            mock_open.__exit__ = MagicMock(return_value=False)
            mock_path.open = MagicMock(return_value=mock_open)
            mock_path.__str__ = MagicMock(return_value="data/ingestion_failures.json")

            import builtins

            with patch.object(builtins, "open", mock_open):
                _print_ingestion_summary(result)

    def test_prints_quota_skipped_when_nonzero(self):
        captured = []
        result = dict(_SUCCESS_RESULT)
        result["documents_skipped_quota"] = 3
        with patch("typer.echo", side_effect=lambda msg="": captured.append(str(msg))):
            _print_ingestion_summary(result)
        output = "\n".join(captured)
        assert "3" in output

    def test_handles_string_errors(self):
        result = dict(_FAILURE_RESULT)
        result["errors"] = ["plain string error"]
        with patch("typer.echo"):
            import builtins

            with patch.object(builtins, "open", MagicMock()):
                _print_ingestion_summary(result)  # Should not raise


# ─── ingest command ───────────────────────────────────────────────────────────


class TestIngestCommand:
    def test_ingest_single_source_success(self, tmp_path):
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli._print_ingestion_summary"),
            patch("src.ingestion.cli.asyncio.run", return_value=_SUCCESS_RESULT),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--source", "https://example.com"])

        assert result.exit_code == 0

    def test_ingest_from_config_success(self, tmp_path):
        p = _seed_yaml(
            tmp_path,
            extra_urls=[
                {
                    "url": "https://example.com",
                    "title": "T",
                    "authority_level": "official",
                    "visa_types": ["chancenkarte"],
                }
            ],
        )
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli._print_ingestion_summary"),
            patch("src.ingestion.cli.asyncio.run", return_value=_SUCCESS_RESULT),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--config", str(p)])

        assert result.exit_code == 0

    def test_ingest_no_docs_exits_1(self, tmp_path):
        p = _seed_yaml(tmp_path, extra_urls=[])
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--config", str(p)])

        assert result.exit_code == 1

    def test_ingest_failure_exits_1(self, tmp_path):
        p = _seed_yaml(
            tmp_path,
            extra_urls=[{"url": "https://example.com", "title": "T", "authority_level": "official", "visa_types": []}],
        )
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli._print_ingestion_summary"),
            patch("src.ingestion.cli.asyncio.run", return_value=_FAILURE_RESULT),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--config", str(p)])

        assert result.exit_code == 1

    def test_ingest_auto_discover_quota_exhausted_exits_2(self):
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli._print_ingestion_summary"),
            patch("src.ingestion.cli.asyncio.run", return_value=_QUOTA_RESULT),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--auto-discover"])

        assert result.exit_code == 2

    def test_ingest_auto_discover_success(self):
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli._print_ingestion_summary"),
            patch("src.ingestion.cli.asyncio.run", return_value=_SUCCESS_RESULT),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--auto-discover"])

        assert result.exit_code == 0

    def test_ingest_auto_discover_failure_exits_1(self):
        mock_pipeline = MagicMock()

        with (
            patch("src.ingestion.cli.get_ingestion_pipeline", return_value=mock_pipeline),
            patch("src.ingestion.cli._print_ingestion_summary"),
            patch("src.ingestion.cli.asyncio.run", return_value=_FAILURE_RESULT),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["ingest", "--auto-discover"])

        assert result.exit_code == 1


# ─── discover command ─────────────────────────────────────────────────────────


class TestDiscoverCommand:
    def test_discover_all_domains(self):
        from src.ingestion.url_discoverer import DiscoveryResult

        mock_result = DiscoveryResult(
            domain="example.com",
            discovered_urls=["https://example.com/page1"],
            from_sitemap=1,
            from_crawling=0,
            filtered_out=2,
        )
        with (
            patch("src.ingestion.cli.asyncio.run", return_value=[mock_result]),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["discover"])

        assert result.exit_code == 0
        assert "example.com" in result.output

    def test_discover_specific_domain(self):
        from src.ingestion.url_discoverer import DiscoveryResult

        mock_result = DiscoveryResult(
            domain="specific.com",
            discovered_urls=["https://specific.com/a", "https://specific.com/b"],
            from_sitemap=2,
        )
        with (
            patch("src.ingestion.cli.asyncio.run", return_value=[mock_result]),
            patch("src.ingestion.cli.logger"),
        ):
            result = runner.invoke(app, ["discover", "--domain", "specific.com"])

        assert result.exit_code == 0
        assert "specific.com" in result.output


# ─── status command ───────────────────────────────────────────────────────────


class TestStatusCommand:
    def test_status_prints_stats(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "ingested_documents": 5,
            "active_chunks": 100,
            "total_ingestion_runs": 3,
        }

        with patch("src.storage.sqlite_state_store.get_state_store", return_value=mock_store):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "ingested_documents" in result.output


# ─── _run_discovery_ingestion ─────────────────────────────────────────────────


class TestRunDiscoveryIngestion:
    @pytest.mark.asyncio
    async def test_returns_no_docs_result_when_crawl_empty(self):
        from src.ingestion.cli import _run_discovery_ingestion

        mock_crawler = AsyncMock()
        mock_crawler.crawl_with_discovery = AsyncMock(return_value=[])
        mock_pipeline = MagicMock()

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            result = await _run_discovery_ingestion(mock_pipeline)

        assert result["success"] is False
        assert result["documents_processed"] == 0

    @pytest.mark.asyncio
    async def test_converts_docs_and_calls_pipeline(self):
        from src.ingestion.cli import _run_discovery_ingestion

        mock_crawler = AsyncMock()
        mock_crawler.crawl_with_discovery = AsyncMock(
            return_value=[
                {
                    "url": "https://example.com",
                    "metadata": {"title": "Page"},
                    "authority_level": "official",
                    "visa_types": ["chancenkarte"],
                },
            ]
        )
        mock_pipeline = AsyncMock()
        mock_pipeline.run_full_ingestion = AsyncMock(return_value=_SUCCESS_RESULT)

        with patch("src.ingestion.crawler.get_crawler", return_value=mock_crawler):
            result = await _run_discovery_ingestion(mock_pipeline)

        assert result["success"] is True
        mock_pipeline.run_full_ingestion.assert_called_once()


# ─── _run_discovery ───────────────────────────────────────────────────────────


class TestRunDiscovery:
    @pytest.mark.asyncio
    async def test_discover_all_when_no_domain(self):
        from src.ingestion.cli import _run_discovery
        from src.ingestion.url_discoverer import DiscoveryResult

        mock_discoverer = AsyncMock()
        mock_discoverer.discover_all = AsyncMock(return_value=[DiscoveryResult(domain="a.com", discovered_urls=[])])
        mock_discoverer.close = AsyncMock()

        with patch("src.ingestion.url_discoverer.get_url_discoverer", return_value=mock_discoverer):
            result = await _run_discovery(domain=None)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_discover_single_domain(self):
        from src.ingestion.cli import _run_discovery
        from src.ingestion.url_discoverer import DiscoveryResult

        mock_discoverer = AsyncMock()
        mock_discoverer.discover_single_domain = AsyncMock(return_value=DiscoveryResult(domain="b.com"))
        mock_discoverer.close = AsyncMock()

        with patch("src.ingestion.url_discoverer.get_url_discoverer", return_value=mock_discoverer):
            result = await _run_discovery(domain="b.com")

        assert result[0].domain == "b.com"
