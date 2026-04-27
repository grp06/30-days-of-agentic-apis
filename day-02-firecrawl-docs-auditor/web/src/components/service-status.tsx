"use client";

import { useEffect, useState } from "react";

import { getStatus } from "@/lib/api";
import type { AuditorStatus } from "@/lib/api";

interface ServiceStatusProps {
  status: AuditorStatus | null;
  onStatusChanged?: (status: AuditorStatus | null) => void;
  showHeader?: boolean;
}

export function ServiceStatusPanel({
  status,
  onStatusChanged,
  showHeader = true,
}: ServiceStatusProps) {
  const [loadState, setLoadState] = useState<"idle" | "refreshing" | "unavailable">("idle");

  useEffect(() => {
    let canceled = false;

    async function refreshStatus() {
      setLoadState("refreshing");
      try {
        const nextStatus = await getStatus();
        if (!canceled) {
          onStatusChanged?.(nextStatus);
          setLoadState("idle");
        }
      } catch {
        if (!canceled) {
          onStatusChanged?.(null);
          setLoadState("unavailable");
        }
      }
    }

    void refreshStatus();
    const interval = window.setInterval(refreshStatus, 15_000);
    return () => {
      canceled = true;
      window.clearInterval(interval);
    };
  }, [onStatusChanged]);

  const services = [
    [
      "Codex app-server",
      status?.codex_app_server.status ?? "api_unavailable",
      status?.codex_app_server.message,
    ],
    [
      "Firecrawl preflight",
      status?.firecrawl.status ?? "api_unavailable",
      status?.firecrawl.message,
    ],
    [
      "Audit engine",
      status?.audit_engine.status ?? "api_unavailable",
      status?.audit_engine.message,
    ],
  ];

  return (
    <section className="panel" aria-label="Service status">
      {showHeader ? (
        <div className="panel-header">
          <p className="eyebrow">Service status</p>
          <span className="muted">
            {status ? `${status.api.host}:${status.api.port}` : formatStatus(loadState)}
          </span>
        </div>
      ) : null}
      <div className="status-list">
        {services.map(([label, value, message]) => (
          <div className="status-row" key={label}>
            <div>
              <span>{label}</span>
              {message ? <small>{message}</small> : null}
            </div>
            <strong>{formatStatus(value ?? "api_unavailable")}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatStatus(status: string): string {
  return status.replaceAll("_", " ");
}
