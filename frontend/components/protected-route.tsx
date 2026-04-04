"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { fetchCurrentUser } from "@/lib/api";
import { clearSession, getStoredSession, type AuthUser } from "@/lib/auth";

type ProtectedRouteProps = {
  children: React.ReactNode;
};

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    async function verify() {
      const session = getStoredSession();
      if (!session) {
        router.replace("/login");
        return;
      }

      try {
        const currentUser = await fetchCurrentUser();
        setUser(currentUser);
      } catch {
        clearSession();
        router.replace("/login?reason=expired");
        return;
      } finally {
        setReady(true);
      }
    }

    void verify();
  }, [router]);

  if (!ready) {
    return (
      <main className="shell">
        <div className="panel section-card">
          <h2>Checking session...</h2>
          <div className="subtle">Verifying clinician access for the ECG workspace.</div>
        </div>
      </main>
    );
  }

  if (!user) {
    return null;
  }

  return <>{children}</>;
}
