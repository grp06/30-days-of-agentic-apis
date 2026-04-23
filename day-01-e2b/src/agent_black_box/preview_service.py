from __future__ import annotations

from .config import Settings
from .fixture_policy import fixture_preview_port
from .preview_execution import build_refresh_context
from .preview_lifecycle import (
    finalize_missing_sandbox_preview,
    finalize_refresh_preview,
    finalize_transient_refresh_error,
)
from .recorder import Recorder
from .run_store import RunStore
from .sandbox_controller import SandboxController


class PreviewService:
    def __init__(
        self,
        settings: Settings,
        run_store: RunStore | None = None,
        sandbox_controller: SandboxController | None = None,
    ) -> None:
        self.settings = settings
        self.run_store = run_store or RunStore(settings.run_root)
        self.sandbox_controller = sandbox_controller or SandboxController(settings)

    async def refresh_preview(self, run_id: str) -> None:
        projected = self.run_store.load_projected_run(run_id)
        if projected is None:
            raise FileNotFoundError(f"Run status not found: {run_id}")
        metadata = projected.metadata
        status = projected.status
        summary = projected.summary
        refresh_context = build_refresh_context(
            metadata=metadata,
            status=status,
            summary=summary,
            evidence=self.run_store.load_preview_evidence(run_id),
            default_preview_port=fixture_preview_port(
                metadata.fixture_name,
                default=self.settings.preview_port,
            ),
        )
        request = refresh_context.request
        if status.state == "running":
            raise RuntimeError("Preview refresh is only available after a run has finished.")
        if not request.allowed:
            raise RuntimeError("Preview refresh is only available for runs with retained sandboxes.")
        recorder = Recorder.open_existing(self.settings.run_root, run_id)
        if metadata.sandbox_id is None:
            next_preview = finalize_missing_sandbox_preview(
                baseline=request.baseline,
                preview_last_error="Run does not have a sandbox id to reconnect.",
            )
            recorder.persist_preview_state(
                preview_state=next_preview.preview_state,
                preview_url=next_preview.preview_url,
                preview_last_error=next_preview.preview_last_error,
                preview_failure_reason=next_preview.preview_failure_reason,
                sandbox_retained=next_preview.sandbox_retained,
            )
            return
        preview_port = request.preview_port or self.settings.preview_port
        try:
            preview_state, preview_url, preview_error = await self.sandbox_controller.refresh_preview(
                metadata.sandbox_id,
                preview_port,
            )
            next_preview = finalize_refresh_preview(
                baseline=request.baseline,
                preview_state=preview_state,
                preview_url=preview_url,
                preview_last_error=preview_error,
                sandbox_retained=True,
            )
            recorder.persist_preview_state(
                preview_state=next_preview.preview_state,
                preview_url=next_preview.preview_url,
                preview_last_error=next_preview.preview_last_error,
                preview_failure_reason=next_preview.preview_failure_reason,
                sandbox_retained=next_preview.sandbox_retained,
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_missing_sandbox_error(exc):
                next_preview = finalize_missing_sandbox_preview(
                    baseline=request.baseline,
                    preview_last_error=str(exc),
                )
                recorder.persist_preview_state(
                    preview_state=next_preview.preview_state,
                    preview_url=next_preview.preview_url,
                    preview_last_error=next_preview.preview_last_error,
                    preview_failure_reason=next_preview.preview_failure_reason,
                    sandbox_retained=next_preview.sandbox_retained,
                )
                return
            next_preview = finalize_transient_refresh_error(
                baseline=request.baseline,
                preview_last_error=str(exc),
            )
            recorder.persist_preview_state(
                preview_state=next_preview.preview_state,
                preview_url=next_preview.preview_url,
                preview_last_error=next_preview.preview_last_error,
                preview_failure_reason=next_preview.preview_failure_reason,
                sandbox_retained=next_preview.sandbox_retained,
            )

    @staticmethod
    def _is_missing_sandbox_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "wasn't found" in message or "not found" in message
