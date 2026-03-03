/* -------------------------------------------------------------------------- *
 *  api.ts – thin wrapper around the NeuraVault FastAPI backend                *
 * -------------------------------------------------------------------------- */

const BASE =
    process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"; // direct to FastAPI backend

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

/* ---- helpers ------------------------------------------------------------ */

async function json<T>(res: Response): Promise<T> {
    if (!res.ok) {
        const body = await res.text();
        throw new Error(body || res.statusText);
    }
    return res.json() as Promise<T>;
}

/* ---- public API --------------------------------------------------------- */

export async function fetchHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE}/api/health`);
    return json<HealthResponse>(res);
}

export async function fetchStarters(): Promise<StarterQuestion[]> {
    const res = await fetch(`${BASE}/api/starters`);
    return json<StarterQuestion[]>(res);
}

export async function queryRAG(question: string): Promise<QueryResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min timeout for slow LLM
    try {
        const res = await fetch(`${BASE}/api/query`, {
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
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const res = await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
    return json<{ status: string; files: string[] }>(res);
}

export async function ingestDocuments(): Promise<IngestResponse> {
    const res = await fetch(`${BASE}/api/ingest`, { method: "POST" });
    return json<IngestResponse>(res);
}

export interface DocFile {
    name: string;
    size_kb: number;
}

export async function fetchDocuments(): Promise<DocFile[]> {
    const res = await fetch(`${BASE}/api/documents`);
    const data = await json<{ files: DocFile[] }>(res);
    return data.files;
}
