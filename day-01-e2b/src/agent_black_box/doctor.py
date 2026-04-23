from __future__ import annotations

from e2b import AsyncSandbox
from pydantic import BaseModel

from .config import Settings
from .model_client import OllamaModelClient


class DoctorReport(BaseModel):
    ollama_auth_ok: bool
    ollama_model_ok: bool
    e2b_auth_ok: bool
    e2b_template_ok: bool
    notes: list[str]


async def run_doctor(settings: Settings) -> DoctorReport:
    notes: list[str] = []
    model_client = OllamaModelClient(settings)
    ollama_auth_ok = False
    ollama_model_ok = False
    e2b_auth_ok = False
    e2b_template_ok = False

    try:
        models = await model_client.list_models()
        ollama_auth_ok = True
        ollama_model_ok = model_client.has_model(settings.ollama_model, models)
        if not ollama_model_ok:
            notes.append(
                f"Ollama model {settings.ollama_model} not returned by /tags; "
                f"available examples: {models[:8]}"
            )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Ollama check failed: {exc}")

    sandbox = None
    try:
        sandbox = await AsyncSandbox.create(
            template=settings.e2b_template,
            timeout=120,
            metadata={"purpose": "doctor", "project": "agent-black-box"},
            envs={"CI": "true"},
            api_key=settings.e2b_api_key,
        )
        e2b_auth_ok = True
        e2b_template_ok = True
    except Exception as exc:  # noqa: BLE001
        notes.append(f"E2B template check failed: {exc}")
        message = str(exc).lower()
        if "template" in message or "not found" in message:
            e2b_auth_ok = True
    finally:
        if sandbox is not None:
            try:
                await sandbox.kill()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Failed to kill doctor sandbox cleanly: {exc}")

    return DoctorReport(
        ollama_auth_ok=ollama_auth_ok,
        ollama_model_ok=ollama_model_ok,
        e2b_auth_ok=e2b_auth_ok,
        e2b_template_ok=e2b_template_ok,
        notes=notes,
    )
