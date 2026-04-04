"use client";

import Link from "next/link";

import { LiveEcgStream } from "@/components/live-ecg-stream";
import { ProtectedRoute } from "@/components/protected-route";
import { SessionControls } from "@/components/session-controls";

function IoTMonitorWorkspace() {
  return (
    <main className="shell">
      <section className="toolbar-row">
        <div>
          <h1 className="page-title">IoT ECG Monitor</h1>
          <div className="subtle">Live ESP32 + AD8232 ingestion plus AI analysis using the existing ECG models.</div>
        </div>
        <div className="stack toolbar-actions">
          <div className="nav-row">
            <Link href="/" className="nav-chip">Dashboard</Link>
            <Link href="/history" className="nav-chip">History</Link>
            <Link href="/alerts" className="nav-chip">Alerts</Link>
            <span className="nav-chip">IoT Monitor</span>
          </div>
          <SessionControls />
        </div>
      </section>

      <LiveEcgStream />
    </main>
  );
}

export default function IoTMonitorPage() {
  return (
    <ProtectedRoute>
      <IoTMonitorWorkspace />
    </ProtectedRoute>
  );
}
