from __future__ import annotations

from pydantic import BaseModel


class DemoCatalogEntry(BaseModel):
    demo_id: str
    title: str
    summary: str
    what_to_notice: str
    fixture_name: str
    default_task: str
    canonical_arena_id: str | None = None


_DEMO_CATALOG: tuple[DemoCatalogEntry, ...] = (
    DemoCatalogEntry(
        demo_id="hello-world-static",
        title="From Hello World to product demo",
        summary="Four agents start from the exact same three-file Hello World app and try to turn it into a focused Agent Black Box landing page.",
        what_to_notice="Because every lane starts from the same tiny static app, the interesting differences come from agent judgment and execution strategy, not framework churn.",
        fixture_name="sample_frontend_task",
        default_task="Turn this tiny static Hello World app into a focused Agent Black Box landing page.",
    ),
)


def load_demo_catalog() -> list[DemoCatalogEntry]:
    return [entry.model_copy(deep=True) for entry in _DEMO_CATALOG]
