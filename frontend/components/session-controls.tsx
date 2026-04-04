"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { clearSession, getStoredSession, type AuthUser } from "@/lib/auth";

export function SessionControls() {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    setUser(getStoredSession()?.user ?? null);
  }, []);

  function signOut() {
    clearSession();
    router.replace("/login");
  }

  return (
    <div className="nav-row session-row">
      {user && <span className="nav-chip">{user.full_name} ({user.role})</span>}
      <button type="button" className="nav-chip nav-chip-button" onClick={signOut}>
        Sign Out
      </button>
    </div>
  );
}
