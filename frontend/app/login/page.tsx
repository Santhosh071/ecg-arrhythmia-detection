"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, useTransition } from "react";

import { login } from "@/lib/api";
import { getStoredSession, saveSession } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [username, setUsername] = useState("clinician");
  const [password, setPassword] = useState("demo123");
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    if (getStoredSession()) {
      router.replace("/");
      return;
    }

    if (searchParams.get("reason") === "expired") {
      setError("Session expired. Sign in again.");
    }
  }, [router, searchParams]);

  function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    startTransition(async () => {
      try {
        const session = await login({ username, password });
        saveSession(session);
        router.replace("/");
      } catch (loginError) {
        setError(loginError instanceof Error ? loginError.message : "Login failed.");
      }
    });
  }

  return (
    <main className="shell auth-shell">
      <section className="panel section-card auth-card">
        <h1 className="page-title auth-title">Clinician Sign In</h1>
        <div className="subtle">Protect the ECG dashboard, history, alerts, reports, and AI assistant behind authenticated access.</div>

        <form onSubmit={onSubmit} className="stack top-gap">
          <label className="field">
            <span>Username</span>
            <input value={username} onChange={(event) => setUsername(event.target.value)} />
          </label>

          <label className="field">
            <span>Password</span>
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>

          {error && <div className="error-box">{error}</div>}

          <button type="submit" className="action-button auth-button">
            {isPending ? "Signing in..." : "Sign In"}
          </button>
        </form>

      </section>
    </main>
  );
}
