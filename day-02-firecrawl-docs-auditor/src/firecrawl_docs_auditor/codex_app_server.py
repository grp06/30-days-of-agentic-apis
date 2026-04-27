from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from .config import Settings


LoginMode = Literal["browser", "device_code"]


class CodexAccount(BaseModel):
    type: str
    email: str | None = None
    plan_type: str | None = None


class CodexAccountStatus(BaseModel):
    status: Literal[
        "available_signed_in",
        "available_login_required",
        "unavailable",
        "error",
    ]
    requires_openai_auth: bool = False
    account: CodexAccount | None = None
    codex_bin_configured: bool = False
    codex_bin_detected: str | None = None
    message: str | None = None


class CodexLoginStartRequest(BaseModel):
    mode: LoginMode = "browser"


class CodexLoginStartResult(BaseModel):
    status: Literal["started", "unavailable", "error"]
    mode: LoginMode
    login_id: str | None = None
    auth_url: str | None = None
    verification_url: str | None = None
    user_code: str | None = None
    message: str | None = None


class CodexLoginCancelRequest(BaseModel):
    login_id: str = Field(min_length=1)


class CodexLoginStatus(BaseModel):
    status: Literal["pending", "succeeded", "failed", "canceled", "unknown"]
    login_id: str
    message: str | None = None


class CodexCancelLoginResult(BaseModel):
    status: Literal["canceled", "not_found", "error"]
    login_id: str
    message: str | None = None


class CodexSmokeTestResult(BaseModel):
    status: Literal["pass", "login_required", "unavailable", "error"]
    message: str
    response: dict[str, Any] | None = None


class CodexStructuredJsonRequest(BaseModel):
    base_instructions: str
    prompt: str
    output_schema: dict[str, Any]
    timeout_seconds: float | None = None


class CodexStructuredJsonResult(BaseModel):
    status: Literal["pass", "login_required", "unavailable", "error"]
    message: str
    response: dict[str, Any] | None = None


class CodexAppServerClient(Protocol):
    async def read_account(self, *, refresh_token: bool = False) -> CodexAccountStatus:
        ...

    async def start_login(self, mode: LoginMode) -> CodexLoginStartResult:
        ...

    async def login_status(self, login_id: str) -> CodexLoginStatus:
        ...

    async def cancel_login(self, login_id: str) -> CodexCancelLoginResult:
        ...

    async def logout(self) -> CodexAccountStatus:
        ...

    async def run_prompt_json_smoke(self) -> CodexSmokeTestResult:
        ...

    async def run_structured_json(
        self,
        request: CodexStructuredJsonRequest,
    ) -> CodexStructuredJsonResult:
        ...


class ManagedCodexAppServerClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._codex_bin = self._resolve_codex_bin(settings.codex_bin)
        self._process: asyncio.subprocess.Process | None = None
        self._next_request_id = 1
        self._lock = asyncio.Lock()
        self._pending_logins: dict[str, CodexLoginStatus] = {}

    async def read_account(self, *, refresh_token: bool = False) -> CodexAccountStatus:
        response = await self._request("account/read", {"refreshToken": refresh_token})
        if response.get("error"):
            return self._account_error(response)
        result = response.get("result") or {}
        account = self._parse_account(result.get("account"))
        if account:
            return CodexAccountStatus(
                status="available_signed_in",
                requires_openai_auth=bool(result.get("requiresOpenaiAuth")),
                account=account,
                codex_bin_configured=bool(self._settings.codex_bin),
                codex_bin_detected=self._codex_bin,
                message="Codex account is signed in.",
            )
        return CodexAccountStatus(
            status="available_login_required",
            requires_openai_auth=bool(result.get("requiresOpenaiAuth", True)),
            codex_bin_configured=bool(self._settings.codex_bin),
            codex_bin_detected=self._codex_bin,
            message="Codex app-server is available; ChatGPT login is required.",
        )

    async def start_login(self, mode: LoginMode) -> CodexLoginStartResult:
        params = {"type": "chatgptDeviceCode" if mode == "device_code" else "chatgpt"}
        response = await self._request("account/login/start", params)
        if response.get("error"):
            return CodexLoginStartResult(
                status="unavailable" if response["error"].get("kind") == "unavailable" else "error",
                mode=mode,
                message=response["error"]["message"],
            )
        result = response.get("result") or {}
        login_id = result.get("loginId")
        if not isinstance(login_id, str):
            return CodexLoginStartResult(
                status="error",
                mode=mode,
                message="Codex app-server did not return a login id.",
            )
        if result.get("type") == "chatgptDeviceCode":
            verification_url = result.get("verificationUrl")
            user_code = result.get("userCode")
            if not isinstance(verification_url, str) or not isinstance(user_code, str):
                return CodexLoginStartResult(
                    status="error",
                    mode="device_code",
                    message="Codex app-server did not return device-code login details.",
                )
            self._pending_logins[login_id] = self._pending_login(login_id)
            return CodexLoginStartResult(
                status="started",
                mode="device_code",
                login_id=login_id,
                verification_url=verification_url,
                user_code=user_code,
                message="Open the verification URL and enter the device code.",
            )
        auth_url = result.get("authUrl")
        if not isinstance(auth_url, str):
            return CodexLoginStartResult(
                status="error",
                mode="browser",
                message="Codex app-server did not return a browser auth URL.",
            )
        self._pending_logins[login_id] = self._pending_login(login_id)
        return CodexLoginStartResult(
            status="started",
            mode="browser",
            login_id=login_id,
            auth_url=auth_url,
            message="Open the auth URL to complete ChatGPT login.",
        )

    @staticmethod
    def _pending_login(login_id: str) -> CodexLoginStatus:
        return CodexLoginStatus(
            status="pending",
            login_id=login_id,
            message="Waiting for ChatGPT login to complete.",
        )

    async def login_status(self, login_id: str) -> CodexLoginStatus:
        await self._drain_notifications()
        return self._pending_logins.get(
            login_id,
            CodexLoginStatus(
                status="unknown",
                login_id=login_id,
                message="No pending login is known for this id.",
            ),
        )

    async def cancel_login(self, login_id: str) -> CodexCancelLoginResult:
        response = await self._request("account/login/cancel", {"loginId": login_id})
        if response.get("error"):
            return CodexCancelLoginResult(
                status="error",
                login_id=login_id,
                message=response["error"]["message"],
            )
        result = response.get("result") or {}
        if result.get("status") == "canceled":
            self._pending_logins[login_id] = CodexLoginStatus(
                status="canceled",
                login_id=login_id,
                message="Login was canceled.",
            )
            return CodexCancelLoginResult(status="canceled", login_id=login_id)
        return CodexCancelLoginResult(
            status="not_found",
            login_id=login_id,
            message="Codex app-server did not know this login id.",
        )

    async def logout(self) -> CodexAccountStatus:
        response = await self._request("account/logout", None)
        if response.get("error"):
            return self._account_error(response)
        return CodexAccountStatus(
            status="available_login_required",
            requires_openai_auth=True,
            codex_bin_configured=bool(self._settings.codex_bin),
            codex_bin_detected=self._codex_bin,
            message="Codex account has been logged out.",
        )

    async def run_prompt_json_smoke(self) -> CodexSmokeTestResult:
        result = await self.run_structured_json(
            CodexStructuredJsonRequest(
                base_instructions="Return only JSON for this connectivity check.",
                prompt='Return {"ok": true, "service": "codex_app_server"} as JSON.',
                output_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ok", "service"],
                    "properties": {
                        "ok": {"type": "boolean"},
                        "service": {"type": "string"},
                    },
                },
            )
        )
        if result.status == "pass":
            return CodexSmokeTestResult(
                status="pass",
                message="Codex app-server returned structured JSON.",
                response=result.response,
            )
        if result.status == "login_required":
            return CodexSmokeTestResult(
                status="login_required",
                message="Sign in with ChatGPT before running the Codex smoke test.",
            )
        return CodexSmokeTestResult(
            status=result.status,
            message=result.message,
            response=result.response,
        )

    async def run_structured_json(
        self,
        request: CodexStructuredJsonRequest,
    ) -> CodexStructuredJsonResult:
        account = await self.read_account()
        if account.status == "available_login_required":
            return CodexStructuredJsonResult(
                status="login_required",
                message="Sign in with ChatGPT before running Codex inference.",
            )
        if account.status in {"unavailable", "error"}:
            return CodexStructuredJsonResult(
                status=account.status,
                message=account.message or "Codex app-server is unavailable.",
            )

        timeout_seconds = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._settings.codex_app_server_timeout_seconds
        )
        thread_response = await self._request(
            "thread/start",
            {
                "ephemeral": True,
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "config": {
                    "web_search": "disabled",
                },
                "experimentalRawEvents": False,
                "persistExtendedHistory": False,
                "baseInstructions": request.base_instructions,
            },
            timeout_seconds=timeout_seconds,
        )
        if thread_response.get("error"):
            return CodexStructuredJsonResult(
                status="error",
                message=thread_response["error"]["message"],
            )
        thread_id = (
            thread_response.get("result", {})
            .get("thread", {})
            .get("id")
        )
        if not isinstance(thread_id, str):
            return CodexStructuredJsonResult(
                status="error",
                message="Codex app-server did not return a thread id.",
            )

        turn_response = await self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "readOnly"},
                "input": [
                    {
                        "type": "text",
                        "text": request.prompt,
                        "text_elements": [],
                    }
                ],
                "outputSchema": request.output_schema,
            },
            timeout_seconds=timeout_seconds * 3,
            collect_turn_for_thread_id=thread_id,
        )
        if turn_response.get("error"):
            return CodexStructuredJsonResult(
                status="error",
                message=turn_response["error"]["message"],
            )
        smoke_payload = turn_response.get("assistant_json")
        if isinstance(smoke_payload, dict):
            return CodexStructuredJsonResult(
                status="pass",
                message="Codex app-server returned structured JSON.",
                response=smoke_payload,
            )
        return CodexStructuredJsonResult(
            status="error",
            message="Codex turn completed without parseable assistant JSON.",
        )

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout_seconds: float | None = None,
        collect_turn_for_thread_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._codex_bin:
            return {
                "error": {
                    "kind": "unavailable",
                    "message": "Codex binary was not found. Set CODEX_BIN if needed.",
                }
            }
        timeout_value = timeout_seconds or self._settings.codex_app_server_timeout_seconds
        async with self._lock:
            try:
                await self._ensure_process(timeout_value)
                request_id = self._next_request_id
                self._next_request_id += 1
                await self._write_message(
                    {"id": request_id, "method": method, "params": params}
                )
                return await self._read_until_response(
                    request_id,
                    timeout_value,
                    collect_turn_for_thread_id=collect_turn_for_thread_id,
                )
            except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as error:
                return {"error": {"kind": "error", "message": str(error)}}

    async def _drain_notifications(self) -> None:
        if not self._process or self._process.returncode is not None:
            return
        async with self._lock:
            while True:
                try:
                    line = await asyncio.wait_for(self._read_stdout_line(), timeout=0.05)
                    message = json.loads(line)
                except asyncio.TimeoutError:
                    return
                except (OSError, ValueError, RuntimeError):
                    return
                if "id" in message:
                    continue
                self._handle_notification(
                    message,
                    collect_turn_for_thread_id=None,
                    turn_id=None,
                )

    async def _ensure_process(self, timeout_seconds: float) -> None:
        if self._process and self._process.returncode is None:
            return
        self._process = await asyncio.create_subprocess_exec(
            self._codex_bin,
            "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._next_request_id = 1
        request_id = self._next_request_id
        self._next_request_id += 1
        await self._write_message(
            {
                "id": request_id,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "firecrawl-docs-auditor",
                        "title": "Firecrawl Docs Auditor",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": False},
                },
            }
        )
        response = await self._read_until_response(request_id, timeout_seconds)
        if response.get("error"):
            raise RuntimeError(response["error"]["message"])
        await self._write_message({"method": "initialized"})

    async def _write_message(self, message: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Codex app-server stdin is unavailable.")
        self._process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        await self._process.stdin.drain()

    async def _read_until_response(
        self,
        request_id: int,
        timeout_seconds: float,
        *,
        collect_turn_for_thread_id: str | None = None,
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        assistant_json: dict[str, Any] | None = None
        turn_id: str | None = None

        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError("Timed out waiting for Codex app-server.")
            line = await asyncio.wait_for(self._read_stdout_line(), timeout=remaining)
            message = json.loads(line)

            if message.get("id") == request_id:
                if "error" in message:
                    return {
                        "error": {
                            "kind": "error",
                            "message": self._safe_rpc_error(message["error"]),
                        }
                    }
                if collect_turn_for_thread_id and assistant_json is not None:
                    return {"result": message.get("result"), "assistant_json": assistant_json}
                if not collect_turn_for_thread_id:
                    return {"result": message.get("result")}
                result_turn_id = message.get("result", {}).get("turn", {}).get("id")
                if isinstance(result_turn_id, str):
                    turn_id = result_turn_id
                continue

            notification_result = self._handle_notification(
                message,
                collect_turn_for_thread_id=collect_turn_for_thread_id,
                turn_id=turn_id,
            )
            if notification_result.get("assistant_json"):
                assistant_json = notification_result["assistant_json"]
            if notification_result.get("turn_done"):
                if assistant_json is not None:
                    return {"result": {}, "assistant_json": assistant_json}
                return {
                    "error": {
                        "kind": "error",
                        "message": notification_result.get(
                            "message", "Codex turn completed without JSON."
                        ),
                    }
                }

    async def _read_stdout_line(self) -> str:
        if not self._process or not self._process.stdout:
            raise RuntimeError("Codex app-server stdout is unavailable.")
        line = await self._process.stdout.readline()
        if not line:
            stderr_tail = await self._read_stderr_tail()
            raise RuntimeError(f"Codex app-server exited. {stderr_tail}".strip())
        return line.decode("utf-8")

    async def _read_stderr_tail(self) -> str:
        if not self._process or not self._process.stderr:
            return ""
        try:
            chunk = await asyncio.wait_for(self._process.stderr.read(4096), timeout=0.1)
        except asyncio.TimeoutError:
            return ""
        return chunk.decode("utf-8", errors="replace").strip()

    def _handle_notification(
        self,
        message: dict[str, Any],
        *,
        collect_turn_for_thread_id: str | None,
        turn_id: str | None,
    ) -> dict[str, Any]:
        method = message.get("method")
        params = message.get("params") or {}

        if method == "account/login/completed":
            login_id = params.get("loginId")
            if isinstance(login_id, str):
                success = bool(params.get("success"))
                self._pending_logins[login_id] = CodexLoginStatus(
                    status="succeeded" if success else "failed",
                    login_id=login_id,
                    message=params.get("error") or ("Login succeeded." if success else "Login failed."),
                )
            return {}

        if (
            method == "item/completed"
            and collect_turn_for_thread_id
            and params.get("threadId") == collect_turn_for_thread_id
            and (turn_id is None or params.get("turnId") == turn_id)
        ):
            item = params.get("item") or {}
            if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                parsed = self._parse_assistant_json(item["text"])
                if parsed is not None:
                    return {"assistant_json": parsed}
            return {}

        if (
            method == "turn/completed"
            and collect_turn_for_thread_id
            and params.get("threadId") == collect_turn_for_thread_id
        ):
            turn = params.get("turn") or {}
            status = turn.get("status")
            if turn_id is not None and turn.get("id") != turn_id:
                return {}
            if status in {"failed", "interrupted"}:
                error = turn.get("error") or {}
                return {
                    "turn_done": True,
                    "message": error.get("message") or f"Codex turn {status}.",
                }
            if status == "completed":
                return {"turn_done": True}
        return {}

    def _account_error(self, response: dict[str, Any]) -> CodexAccountStatus:
        error = response["error"]
        return CodexAccountStatus(
            status="unavailable" if error.get("kind") == "unavailable" else "error",
            codex_bin_configured=bool(self._settings.codex_bin),
            codex_bin_detected=self._codex_bin,
            message=error["message"],
        )

    @staticmethod
    def _parse_account(raw_account: object) -> CodexAccount | None:
        if not isinstance(raw_account, dict):
            return None
        return CodexAccount(
            type=str(raw_account.get("type") or "unknown"),
            email=raw_account.get("email"),
            plan_type=raw_account.get("planType"),
        )

    @staticmethod
    def _parse_assistant_json(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = [line for line in stripped.splitlines() if not line.startswith("```")]
            stripped = "\n".join(lines).strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _safe_rpc_error(error: object) -> str:
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        return "Codex app-server returned an error."

    @staticmethod
    def _resolve_codex_bin(configured_bin: str | None) -> str | None:
        if configured_bin:
            configured_path = Path(configured_bin).expanduser()
            if configured_path.name != configured_bin:
                return str(configured_path) if configured_path.exists() else None
            return shutil.which(configured_bin)
        path_bin = shutil.which("codex")
        if path_bin:
            return path_bin
        app_bin = Path("/Applications/Codex.app/Contents/Resources/codex")
        if app_bin.exists():
            return str(app_bin)
        return None
