"use client";

import { useCallback, useEffect, useState } from "react";

import {
  cancelCodexLogin,
  getCodexAccount,
  getCodexLoginStatus,
  logoutCodex,
  runCodexSmokeTest,
  startCodexLogin,
  type CodexAccountStatus,
  type CodexLoginStartResult,
  type CodexLoginStatus,
  type CodexSmokeTestResult,
} from "@/lib/api";

type LoadState = "idle" | "loading" | "error";

interface CodexAuthPanelProps {
  showHeader?: boolean;
  variant?: "panel" | "header";
}

export function CodexAuthPanel({ showHeader = true, variant = "panel" }: CodexAuthPanelProps) {
  const [account, setAccount] = useState<CodexAccountStatus | null>(null);
  const [login, setLogin] = useState<CodexLoginStartResult | null>(null);
  const [loginStatus, setLoginStatus] = useState<CodexLoginStatus | null>(null);
  const [smoke, setSmoke] = useState<CodexSmokeTestResult | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [action, setAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshAccount = useCallback(async () => {
    setLoadState("loading");
    setError(null);
    try {
      setAccount(await getCodexAccount());
      setLoadState("idle");
    } catch (caught) {
      setError(errorMessage(caught));
      setLoadState("error");
    }
  }, []);

  useEffect(() => {
    let active = true;
    getCodexAccount()
      .then((nextAccount) => {
        if (!active) {
          return;
        }
        setAccount(nextAccount);
        setLoadState("idle");
      })
      .catch((caught: unknown) => {
        if (!active) {
          return;
        }
        setError(errorMessage(caught));
        setLoadState("error");
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!login?.login_id || (loginStatus && loginStatus.status !== "pending")) {
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        const nextStatus = await getCodexLoginStatus(login.login_id as string);
        setLoginStatus(nextStatus);
        if (nextStatus.status === "succeeded") {
          await refreshAccount();
        }
      } catch (caught) {
        setError(errorMessage(caught));
      }
    }, 2500);
    return () => window.clearTimeout(timer);
  }, [login, loginStatus, refreshAccount]);

  async function startLogin(mode: "browser" | "device_code") {
    setAction(mode);
    setError(null);
    setSmoke(null);
    try {
      const result = await startCodexLogin(mode);
      if (result.status !== "started") {
        setLogin(null);
        setLoginStatus(null);
        setError(result.message ?? "Codex login could not be started.");
        return;
      }
      setLogin(result);
      if (result.login_id) {
        setLoginStatus({
          status: "pending",
          login_id: result.login_id,
          message: result.message,
        });
      }
      if (result.auth_url) {
        window.open(result.auth_url, "_blank", "noopener,noreferrer");
      }
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setAction(null);
    }
  }

  async function cancelLogin() {
    if (!login?.login_id) {
      return;
    }
    setAction("cancel");
    setError(null);
    try {
      const result = await cancelCodexLogin(login.login_id);
      setLoginStatus({
        status: result.status === "canceled" ? "canceled" : "unknown",
        login_id: login.login_id,
        message: result.message,
      });
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setAction(null);
    }
  }

  async function logout() {
    setAction("logout");
    setError(null);
    setSmoke(null);
    try {
      setAccount(await logoutCodex());
      setLogin(null);
      setLoginStatus(null);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setAction(null);
    }
  }

  async function smokeTest() {
    setAction("smoke");
    setError(null);
    try {
      setSmoke(await runCodexSmokeTest());
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setAction(null);
    }
  }

  const signedIn = account?.status === "available_signed_in";
  const loginPending = loginStatus?.status === "pending";
  const canStartLogin =
    !signedIn &&
    !loginPending &&
    account?.status !== "unavailable" &&
    account?.status !== "error";
  const summaryMessage = signedIn
    ? null
    : account?.message ?? "Checking Codex app-server.";
  const statusLabel = formatStatus(account?.status ?? loadState);

  const authBody = (
    <>
      <div className="auth-summary">
        <strong>{accountLabel(account)}</strong>
        {summaryMessage ? <span>{summaryMessage}</span> : null}
      </div>

      {login?.verification_url ? (
        <div className="auth-device">
          <a href={login.verification_url} target="_blank" rel="noreferrer">
            {login.verification_url}
          </a>
          <strong>{login.user_code}</strong>
        </div>
      ) : null}

      {loginStatus ? (
        <div className="auth-status-row">
          <span>{formatStatus(loginStatus.status)}</span>
          <small>{loginStatus.message ?? loginStatus.login_id}</small>
        </div>
      ) : null}

      <div className={`auth-actions ${signedIn ? "resolved" : ""}`}>
        {canStartLogin ? (
          <>
            <button type="button" onClick={() => startLogin("browser")} disabled={!!action}>
              Sign in
            </button>
            <button type="button" onClick={() => startLogin("device_code")} disabled={!!action}>
              Device code
            </button>
          </>
        ) : null}
        {loginPending ? (
          <button type="button" onClick={cancelLogin} disabled={!!action || !login?.login_id}>
            Cancel login
          </button>
        ) : null}
        {signedIn ? (
          <>
            <button type="button" className="text-button" onClick={smokeTest} disabled={!!action}>
              Run smoke test
            </button>
            <button type="button" onClick={logout} disabled={!!action}>
              Logout
            </button>
          </>
        ) : null}
      </div>

      {!signedIn ? (
        <button type="button" className="secondary-button" onClick={smokeTest} disabled={!!action}>
          Run smoke test
        </button>
      ) : null}

      {smoke ? (
        <div className={`auth-result ${smoke.status}`}>
          <strong>{formatStatus(smoke.status)}</strong>
          <span>{smoke.message}</span>
        </div>
      ) : null}

      {error ? <p className="auth-error">{error}</p> : null}
    </>
  );

  if (variant === "header") {
    return (
      <details className="account-menu">
        <summary>
          <span>Codex</span>
          <strong>{accountLabel(account)}</strong>
        </summary>
        <div className="account-menu-body">
          <span className={`pill ${account?.status ?? "loading"}`}>{statusLabel}</span>
          {authBody}
        </div>
      </details>
    );
  }

  return (
    <section className="panel auth-panel" aria-label="Codex authentication">
      {showHeader ? (
        <div className="panel-header">
          <p className="eyebrow">Codex auth</p>
          <span className={`pill ${account?.status ?? "loading"}`}>{statusLabel}</span>
        </div>
      ) : null}
      {authBody}
    </section>
  );
}

function accountLabel(account: CodexAccountStatus | null): string {
  if (!account) {
    return "Checking account";
  }
  if (account.account?.email) {
    return account.account.email;
  }
  return formatStatus(account.status);
}

function formatStatus(status: string): string {
  return status.replaceAll("_", " ");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Codex auth request failed.";
}
