/* -------------------------------------------------------------------------- *
 *  api.ts – thin wrapper around the NeuraVault FastAPI backend                *
 * -------------------------------------------------------------------------- */

const BASE =
    process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"; // direct to FastAPI backend

/* -------------------------------------------------------------------------- *
 *  Auth token management                                                      *
 * -------------------------------------------------------------------------- */

const TOKEN_KEY = "neuravault_token";
const USER_KEY = "neuravault_user";

export function getToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

export function getStoredUser(): AuthUser | null {
    if (typeof window === "undefined") return null;
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try { return JSON.parse(raw); } catch { return null; }
}

export function setStoredUser(user: AuthUser): void {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export interface AuthUser {
    username: string;
    role: string;
}

export interface LoginResponse {
    access_token: string;
    token_type: string;
    expires_in: number;
    role: string;
    username: string;
}

/* -------------------------------------------------------------------------- *
 *  CSRF token helper                                                          *
 * -------------------------------------------------------------------------- */
function getCsrfToken(): string {
    if (typeof document === "undefined") return "";
    const match = document.cookie.match(/neuravault_csrf=([^;]+)/);
    return match ? match[1] : "";
}

/** Build common headers including auth token and CSRF token */
function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = { ...extra };
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;
    return headers;
}

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

/* ---- New feature types -------------------------------------------------- */

export interface TimelineEvent {
    date: string;
    event: string;
    source: string;
    confidence: string;
}

export interface TimelineResponse {
    events: TimelineEvent[];
    has_timeline: boolean;
    total_chunks_analyzed: number;
    message: string;
}

export interface AnomalyAlert {
    type: string;
    severity: string;
    title: string;
    description: string;
    source: string;
    recommendation: string;
}

export interface AnomalyResponse {
    alerts: AnomalyAlert[];
    total_alerts: number;
    severity_counts: Record<string, number>;
    chunks_analyzed: number;
    message: string;
}

export interface GraphNode {
    id: string;
    type: string;
    source: string;
}

export interface GraphEdge {
    from: string;
    to: string;
    relationship: string;
    strength: string;
    source: string;
}

export interface EvidenceLinkResponse {
    nodes: GraphNode[];
    edges: GraphEdge[];
    total_nodes: number;
    total_edges: number;
    chunks_analyzed: number;
    message: string;
}

export interface PromptTemplate {
    id: string;
    name: string;
    description: string;
    prompt: string;
    is_default: boolean;
}

export interface ExplainableResponse {
    answer: string;
    sources: SourceInfo[];
    reasoning_mode: boolean;
}

export interface DailyCount {
    date: string;
    count: number;
}

export interface UsageFeatureCount {
    feature: string;
    event_type: string;
    count: number;
}

export interface TemplateUsageCount {
    template_id: string;
    count: number;
}

export interface UsageSummary {
    total_usage_events: number;
    successful_usage_events: number;
    failed_usage_events: number;
    unique_users: number;
    total_logins: number;
    total_registrations: number;
    total_queries: number;
    total_uploads: number;
    total_ingestions: number;
    total_feature_invocations: number;
    daily_queries: DailyCount[];
    daily_active_users: DailyCount[];
    feature_usage: UsageFeatureCount[];
    template_usage: TemplateUsageCount[];
}

export interface DailyQuality {
    date: string;
    queries: number;
    citation_rate: number;
    no_answer_rate: number;
}

export interface RagQualitySummary {
    total_queries_analyzed: number;
    queries_with_citations: number;
    citation_coverage_rate: number;
    average_sources_per_query: number;
    no_answer_rate: number;
    average_answer_length_chars: number;
    average_relevance_score: number | null;
    daily_quality: DailyQuality[];
}

export interface AnalyticsSummary {
    window_days: number;
    generated_at: string;
    usage: UsageSummary;
    rag_quality: RagQualitySummary;
}

export interface ChatMessageExport {
    role: string;
    content: string;
    sources?: SourceInfo[];
}

/* ---- helpers ------------------------------------------------------------ */

async function json<T>(res: Response): Promise<T> {
    if (!res.ok) {
        if (res.status === 401) {
            clearToken();
            if (typeof window !== "undefined") {
                window.dispatchEvent(new Event("neuravault:unauthorized"));
            }
        }
        const body = await res.text();
        throw new Error(formatApiError(body, res.statusText));
    }
    return res.json() as Promise<T>;
}

function formatApiError(rawBody: string, fallback: string): string {
    if (!rawBody) return fallback || "Request failed";

    try {
        const parsed = JSON.parse(rawBody);
        const detail = parsed?.detail;

        if (Array.isArray(detail)) {
            const messages = detail
                .map((item: unknown) => {
                    if (typeof item === "string") return item;
                    if (item && typeof item === "object" && "msg" in item) {
                        return String((item as { msg?: unknown }).msg ?? "");
                    }
                    return "";
                })
                .filter(Boolean);

            if (messages.length > 0) {
                return messages.join(" | ");
            }
        }

        if (typeof detail === "string") {
            return detail;
        }

        return rawBody;
    } catch {
        return rawBody;
    }
}

/* ---- Auth API ----------------------------------------------------------- */

export async function login(username: string, password: string): Promise<LoginResponse> {
    const res = await fetch(`${BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ username, password }),
    });
    const data = await json<LoginResponse>(res);
    setToken(data.access_token);
    setStoredUser({ username: data.username, role: data.role });
    return data;
}

export async function register(
    username: string,
    password: string,
    role: string = "analyst"
): Promise<LoginResponse> {
    const res = await fetch(`${BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ username, password, role }),
    });
    const data = await json<LoginResponse>(res);
    setToken(data.access_token);
    setStoredUser({ username: data.username, role: data.role });
    return data;
}

export function logout(): void {
    clearToken();
}

export async function fetchCurrentUser(): Promise<AuthUser> {
    const res = await fetch(`${BASE}/api/auth/me`, {
        headers: authHeaders(),
        credentials: "include",
    });
    return json<AuthUser>(res);
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

export async function queryRAG(question: string): Promise<QueryResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min timeout for slow LLM
    try {
        const res = await fetch(`${BASE}/api/query`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ question }),
            signal: controller.signal,
            credentials: "include",
        });
        return json<QueryResponse>(res);
    } finally {
        clearTimeout(timeout);
    }
}

export async function uploadFiles(
    files: FileList | File[]
): Promise<{ status: string; files: string[] }> {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const hdrs = authHeaders();
    const res = await fetch(`${BASE}/api/upload`, {
        method: "POST",
        headers: hdrs,
        body: form,
        credentials: "include",
    });
    return json<{ status: string; files: string[] }>(res);
}

export async function ingestDocuments(): Promise<IngestResponse> {
    const res = await fetch(`${BASE}/api/ingest`, {
        method: "POST",
        headers: authHeaders(),
        credentials: "include",
    });
    return json<IngestResponse>(res);
}

export interface DocFile {
    name: string;
    size_kb: number;
}

export async function fetchDocuments(): Promise<DocFile[]> {
    const res = await fetch(`${BASE}/api/documents`, {
        headers: authHeaders(),
        credentials: "include",
    });
    const data = await json<{ files: DocFile[] }>(res);
    return data.files;
}

/* ---- Feature: Timeline -------------------------------------------------- */

export async function fetchTimeline(): Promise<TimelineResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);
    try {
        const res = await fetch(`${BASE}/api/timeline`, {
            method: "POST",
            headers: authHeaders(),
            signal: controller.signal,
            credentials: "include",
        });
        return json<TimelineResponse>(res);
    } finally {
        clearTimeout(timeout);
    }
}

/* ---- Feature: Anomaly Detection ----------------------------------------- */

export async function fetchAnomalies(): Promise<AnomalyResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);
    try {
        const res = await fetch(`${BASE}/api/anomalies`, {
            method: "POST",
            headers: authHeaders(),
            signal: controller.signal,
            credentials: "include",
        });
        return json<AnomalyResponse>(res);
    } finally {
        clearTimeout(timeout);
    }
}

/* ---- Feature: Chat Export ----------------------------------------------- */

export async function exportChat(
    messages: ChatMessageExport[],
    format: "pdf" | "docx" = "pdf",
    title = "NeuraVault Chat Export"
): Promise<Blob> {
    const res = await fetch(`${BASE}/api/export`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ messages, format, title }),
        credentials: "include",
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(body || res.statusText);
    }
    return res.blob();
}

/* ---- Feature: Evidence Linking ------------------------------------------ */

export async function fetchEvidenceLinks(): Promise<EvidenceLinkResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);
    try {
        const res = await fetch(`${BASE}/api/evidence-links`, {
            method: "POST",
            headers: authHeaders(),
            signal: controller.signal,
            credentials: "include",
        });
        return json<EvidenceLinkResponse>(res);
    } finally {
        clearTimeout(timeout);
    }
}

/* ---- Feature: Prompt Templates ------------------------------------------ */

export async function fetchTemplates(): Promise<PromptTemplate[]> {
    const res = await fetch(`${BASE}/api/templates`, {
        headers: authHeaders(),
        credentials: "include",
    });
    const data = await json<{ templates: PromptTemplate[] }>(res);
    return data.templates;
}

export async function createTemplate(
    template: Omit<PromptTemplate, "is_default">
): Promise<PromptTemplate> {
    const res = await fetch(`${BASE}/api/templates`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(template),
        credentials: "include",
    });
    const data = await json<{ status: string; template: PromptTemplate }>(res);
    return data.template;
}

export async function deleteTemplate(
    templateId: string
): Promise<void> {
    const res = await fetch(`${BASE}/api/templates/${templateId}`, {
        method: "DELETE",
        headers: authHeaders(),
        credentials: "include",
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(body || res.statusText);
    }
}

/* ---- Feature: Explainable AI + Streaming -------------------------------- */

export async function queryExplainable(
    question: string,
    templateId?: string,
): Promise<ExplainableResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);
    try {
        const res = await fetch(`${BASE}/api/query/explain`, {
            method: "POST",
            headers: authHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({
                question,
                template_id: templateId ?? null,
                stream: false,
            }),
            signal: controller.signal,
            credentials: "include",
        });
        return json<ExplainableResponse>(res);
    } finally {
        clearTimeout(timeout);
    }
}

export async function queryExplainableStream(
    question: string,
    templateId?: string,
    onToken?: (token: string) => void,
    onSources?: (sources: SourceInfo[]) => void,
    onDone?: () => void,
    onError?: (error: string) => void,
): Promise<void> {
    const res = await fetch(`${BASE}/api/query/explain`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
            question,
            template_id: templateId ?? null,
            stream: true,
        }),
        credentials: "include",
    });

    if (!res.ok) {
        const body = await res.text();
        throw new Error(body || res.statusText);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const event = JSON.parse(line);
                if (event.type === "token" && onToken) {
                    onToken(event.data);
                } else if (event.type === "sources" && onSources) {
                    onSources(event.data);
                } else if (event.type === "done" && onDone) {
                    onDone();
                } else if (event.type === "error" && onError) {
                    onError(event.data);
                }
            } catch {
                // skip malformed lines
            }
        }
    }
}

/* ---- Analytics ---------------------------------------------------------- */

export async function fetchAnalyticsSummary(days = 30): Promise<AnalyticsSummary> {
    const res = await fetch(`${BASE}/api/analytics/summary?days=${days}`, {
        headers: authHeaders(),
        credentials: "include",
    });
    return json<AnalyticsSummary>(res);
}
