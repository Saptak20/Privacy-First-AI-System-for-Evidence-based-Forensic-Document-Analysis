/* -------------------------------------------------------------------------- *
 *  api.ts – thin wrapper around the NeuraVault FastAPI backend                *
 *                                                                             *
 *  Security features:                                                         *
 *    ✔ JWT bearer token sent on every protected request                       *
 *    ✔ CSRF double-submit cookie pattern                                      *
 *    ✔ Auto-refresh of expired access tokens                                  *
 * -------------------------------------------------------------------------- */

const BASE =
    process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"; // direct to FastAPI backend

/* ---- types -------------------------------------------------------------- */

export interface SourceInfo {
    filename: string;
    document_type: string;
    excerpt: string;
}

export interface QueryResponse {
    answer: string;
    sources: SourceInfo[];
}

export interface StarterQuestion {
    label: string;
    message: string;
}

export interface HealthResponse {
    status: string;
    engine_ready: boolean;
    vector_db_exists: boolean;
    model_name: string;
}

export interface IngestResponse {
    status: string;
    documents_processed: number;
    chunks_created: number;
}

export interface TokenResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    username: string;
    role: string;
}

export interface UserProfile {
    id: number;
    username: string;
    email: string;
    role: string;
    is_active: boolean;
}

/* ---- token storage ------------------------------------------------------ */

const TOKEN_KEY = "nv_access_token";
const REFRESH_KEY = "nv_refresh_token";
const USER_KEY = "nv_user";

export function saveAuth(data: TokenResponse): void {
    localStorage.setItem(TOKEN_KEY, data.access_token);
    localStorage.setItem(REFRESH_KEY, data.refresh_token);
    localStorage.setItem(USER_KEY, JSON.stringify({ username: data.username, role: data.role }));
}

export function getAccessToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
}

export function getRefreshToken(): string | null {
    return localStorage.getItem(REFRESH_KEY);
}

export function getStoredUser(): { username: string; role: string } | null {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
}

export function clearAuth(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
}

export function isAuthenticated(): boolean {
    return !!getAccessToken();
}

/* ---- CSRF helper -------------------------------------------------------- */

function getCsrfToken(): string | null {
    const match = document.cookie
        .split("; ")
        .find((row) => row.startsWith("csrf_token="));
    return match ? match.split("=")[1] : null;
}

/** Fetch a fresh CSRF token from the server (sets cookie automatically). */
export async function ensureCsrfToken(): Promise<string> {
    const existing = getCsrfToken();
    if (existing) return existing;
    const res = await fetch(`${BASE}/api/csrf-token`, { credentials: "include" });
    const data = await res.json();
    return data.csrf_token as string;
}

/* ---- helpers ------------------------------------------------------------ */

async function json<T>(res: Response): Promise<T> {
    if (!res.ok) {
        const body = await res.text();
        throw new Error(body || res.statusText);
    }
    return res.json() as Promise<T>;
}

/** Build headers with auth + CSRF tokens for mutating requests. */
function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = { ...extra };
    const token = getAccessToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;
    return headers;
}

/** Attempt a single token refresh. Returns true on success. */
async function tryRefresh(): Promise<boolean> {
    const rt = getRefreshToken();
    if (!rt) return false;
    try {
        const res = await fetch(`${BASE}/api/auth/refresh`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refresh_token: rt }),
            credentials: "include",
        });
        if (!res.ok) return false;
        const data: TokenResponse = await res.json();
        saveAuth(data);
        return true;
    } catch {
        return false;
    }
}

/**
 * Wrapper around fetch that auto-retries once with a refreshed token on 401.
 */
async function authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
    const doFetch = () =>
        fetch(url, {
            ...init,
            credentials: "include",
            headers: {
                ...authHeaders(),
                ...(init.headers as Record<string, string>),
            },
        });

    let res = await doFetch();
    if (res.status === 401) {
        const refreshed = await tryRefresh();
        if (refreshed) res = await doFetch();
    }
    return res;
}

/* ---- Auth API ----------------------------------------------------------- */

export async function register(
    username: string,
    email: string,
    password: string,
): Promise<TokenResponse> {
    await ensureCsrfToken();
    const res = await fetch(`${BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
        credentials: "include",
    });
    const data = await json<TokenResponse>(res);
    saveAuth(data);
    return data;
}

export async function login(
    username: string,
    password: string,
): Promise<TokenResponse> {
    await ensureCsrfToken();
    const res = await fetch(`${BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
        credentials: "include",
    });
    const data = await json<TokenResponse>(res);
    saveAuth(data);
    return data;
}

export function logout(): void {
    clearAuth();
}

export async function fetchProfile(): Promise<UserProfile> {
    const res = await authedFetch(`${BASE}/api/auth/me`);
    return json<UserProfile>(res);
}

/* ---- public API --------------------------------------------------------- */

export async function fetchHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE}/api/health`, { credentials: "include" });
    return json<HealthResponse>(res);
}

export async function fetchStarters(): Promise<StarterQuestion[]> {
    const res = await fetch(`${BASE}/api/starters`, { credentials: "include" });
    return json<StarterQuestion[]>(res);
}

/* ---- protected API ------------------------------------------------------ */

export async function queryRAG(question: string): Promise<QueryResponse> {
    await ensureCsrfToken();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min timeout for slow LLM
    try {
        const res = await authedFetch(`${BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
            signal: controller.signal,
        });
        return json<QueryResponse>(res);
    } finally {
        clearTimeout(timeout);
    }
}

export async function uploadFiles(
    files: FileList | File[]
): Promise<{ status: string; files: string[] }> {
    await ensureCsrfToken();
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const res = await authedFetch(`${BASE}/api/upload`, {
        method: "POST",
        body: form,
    });
    return json<{ status: string; files: string[] }>(res);
}

export async function ingestDocuments(): Promise<IngestResponse> {
    await ensureCsrfToken();
    const res = await authedFetch(`${BASE}/api/ingest`, { method: "POST" });
    return json<IngestResponse>(res);
}

export interface DocFile {
    name: string;
    size_kb: number;
}

export async function fetchDocuments(): Promise<DocFile[]> {
    const res = await authedFetch(`${BASE}/api/documents`);
    const data = await json<{ files: DocFile[] }>(res);
    return data.files;
}
