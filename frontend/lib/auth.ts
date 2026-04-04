export type AuthUser = {
  username: string;
  full_name: string;
  role: string;
};

export type AuthSession = {
  access_token: string;
  token_type: string;
  user: AuthUser;
};

const STORAGE_KEY = "ecg_auth_session";
export const SESSION_EXPIRED_EVENT = "ecg:session-expired";

export function getStoredSession(): AuthSession | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthSession;
  } catch {
    return null;
  }
}

export function getStoredToken(): string | null {
  return getStoredSession()?.access_token ?? null;
}

export function saveSession(session: AuthSession) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export function expireSession() {
  if (typeof window === "undefined") return;
  clearSession();
  window.dispatchEvent(new CustomEvent(SESSION_EXPIRED_EVENT));
}
