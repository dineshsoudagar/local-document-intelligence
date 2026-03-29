from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.api.routes_health import _build_health_details_payload, _is_backend_ready
from src.app.paths import AppPaths
from src.app.runtime_state import SetupStatus
from src.indexing.index_service import IndexService


class _FlakyParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        if self.calls == 1:
            raise ValueError(f"Input document {path.name} is not valid.")
        return [{"doc_id": doc_id, "path": str(path)}]


class _BrokenParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        raise ValueError("Synthetic permanent parser failure")


class _WarmupParser:
    def __init__(self) -> None:
        self.calls = 0
        self.paths: list[Path] = []

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        self.paths.append(path)
        return []


class _FailingWarmupParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, *, doc_id: str) -> list[object]:
        self.calls += 1
        raise RuntimeError("Synthetic warmup failure")


class _FakeRuntimeController:
    def __init__(self, diagnostics: dict[str, str | bool | None]) -> None:
        self._diagnostics = diagnostics

    def diagnostics(self) -> dict[str, str | bool | None]:
        return dict(self._diagnostics)


def _build_service(parser: object) -> IndexService:
    service = object.__new__(IndexService)
    service._parser = parser
    service._parser_warmed_up = False
    service._parser_warmup_ran_in_process = False
    service._parser_warmup_started_at = None
    service._parser_warmup_completed_at = None
    service._parser_warmup_error = None
    return service


class IndexServiceRetryTests(unittest.TestCase):
    def test_parse_chunks_retries_transient_docling_validation_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

            parser = _FlakyParser()
            service = _build_service(parser)

            chunks = service._parse_chunks(pdf_path, "doc_retry")

            self.assertEqual(parser.calls, 2)
            self.assertEqual(
                chunks,
                [{"doc_id": "doc_retry", "path": str(pdf_path.resolve())}],
            )

    def test_parse_chunks_does_not_retry_permanent_parser_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%test\n")

            parser = _BrokenParser()
            service = _build_service(parser)

            with self.assertRaisesRegex(ValueError, "Synthetic permanent parser failure"):
                service._parse_chunks(pdf_path, "doc_fail")

            self.assertEqual(parser.calls, 1)

    def test_warm_up_parser_primes_once_and_updates_snapshot(self) -> None:
        parser = _WarmupParser()
        service = _build_service(parser)

        service.warm_up_parser()
        service.warm_up_parser()

        snapshot = service.parser_warmup_snapshot()
        self.assertEqual(parser.calls, 1)
        self.assertTrue(service._parser_warmed_up)
        self.assertTrue(snapshot["parser_warmup_ran_in_process"])
        self.assertTrue(snapshot["parser_warmup_completed"])
        self.assertIsNotNone(snapshot["parser_warmup_started_at"])
        self.assertIsNotNone(snapshot["parser_warmup_completed_at"])
        self.assertIsNone(snapshot["parser_warmup_error"])
        self.assertEqual(parser.paths[0].suffix.lower(), ".pdf")

    def test_warm_up_parser_records_failure_state(self) -> None:
        parser = _FailingWarmupParser()
        service = _build_service(parser)

        with self.assertRaisesRegex(RuntimeError, "Synthetic warmup failure"):
            service.warm_up_parser()

        snapshot = service.parser_warmup_snapshot()
        self.assertEqual(parser.calls, 1)
        self.assertFalse(snapshot["parser_warmup_completed"])
        self.assertTrue(snapshot["parser_warmup_ran_in_process"])
        self.assertIsNotNone(snapshot["parser_warmup_started_at"])
        self.assertIsNone(snapshot["parser_warmup_completed_at"])
        self.assertEqual(snapshot["parser_warmup_error"], "Synthetic warmup failure")


class RuntimeHealthDiagnosticsTests(unittest.TestCase):
    def test_health_details_payload_includes_runtime_and_log_diagnostics(self) -> None:
        paths = AppPaths(
            app_root=Path(r"C:\app-root"),
            code_root=Path(r"C:\code-root"),
        )
        setup_status = SetupStatus(install_state="ready")
        controller = _FakeRuntimeController(
            {
                "runtime_initialized": True,
                "runtime_last_error": None,
                "parser_warmup_ran_in_process": True,
                "parser_warmup_started_at": "2026-03-29T12:00:00+00:00",
                "parser_warmup_completed_at": "2026-03-29T12:00:02+00:00",
                "parser_warmup_completed": True,
                "parser_warmup_error": None,
            }
        )

        payload = _build_health_details_payload(
            paths=paths,
            setup_status=setup_status,
            runtime_controller=controller,
            runtime_mode="managed_subprocess",
            launcher_log_path=r"C:\app-root\logs\launcher.log",
            backend_log_path=r"C:\app-root\logs\backend.log",
        )

        self.assertEqual(payload["runtime_mode"], "managed_subprocess")
        self.assertEqual(payload["launcher_log_path"], r"C:\app-root\logs\launcher.log")
        self.assertEqual(payload["backend_log_path"], r"C:\app-root\logs\backend.log")
        self.assertTrue(payload["runtime_initialized"])
        self.assertTrue(payload["parser_warmup_completed"])

    def test_backend_ready_requires_completed_parser_warmup(self) -> None:
        base_payload = {
            "app_root": r"C:\app-root",
            "launcher_log_path": r"C:\app-root\logs\launcher.log",
            "backend_log_path": r"C:\app-root\logs\backend.log",
            "install_state": "ready",
            "runtime_initialized": True,
            "parser_warmup_completed": False,
        }

        self.assertFalse(_is_backend_ready(base_payload))

        ready_payload = dict(base_payload)
        ready_payload["parser_warmup_completed"] = True
        self.assertTrue(_is_backend_ready(ready_payload))

        missing_log_payload = dict(ready_payload)
        missing_log_payload["launcher_log_path"] = ""
        self.assertFalse(_is_backend_ready(missing_log_payload))


if __name__ == "__main__":
    unittest.main()
