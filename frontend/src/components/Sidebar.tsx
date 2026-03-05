"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
    Upload,
    Database,
    FileText,
    X,
    FolderUp,
    RefreshCw,
    CheckCircle2,
    AlertCircle,
    File,
    Loader2,
} from "lucide-react";
import {
    uploadFiles,
    ingestDocuments,
    fetchDocuments,
    type DocFile,
    type IngestResponse,
} from "@/lib/api";

interface Props {
    open: boolean;
    onClose: () => void;
    engineReady: boolean | null;
    onEngineReady: () => void;          // notify parent that engine is now ready
}

export default function Sidebar({ open, onClose, engineReady, onEngineReady }: Props) {
    const [docs, setDocs] = useState<DocFile[]>([]);
    const [uploading, setUploading] = useState(false);
    const [ingesting, setIngesting] = useState(false);
    const [dragOver, setDragOver] = useState(false);
    const [statusMsg, setStatusMsg] = useState<{ text: string; ok: boolean } | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    /* load document list on mount + after changes */
    const refreshDocs = useCallback(() => {
        fetchDocuments().then(setDocs).catch(() => { });
    }, []);

    useEffect(() => { refreshDocs(); }, [refreshDocs]);

    /* ---- upload handler --------------------------------------------------- */
    const handleFiles = async (files: FileList | File[]) => {
        if (!files || (files as FileList).length === 0) return;
        setUploading(true);
        setStatusMsg(null);
        try {
            await uploadFiles(files);
            setStatusMsg({ text: `${(files as FileList).length} file(s) uploaded`, ok: true });
            refreshDocs();
        } catch (err: any) {
            setStatusMsg({ text: err.message ?? "Upload failed", ok: false });
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    /* ---- ingest handler --------------------------------------------------- */
    const handleIngest = async () => {
        setIngesting(true);
        setStatusMsg(null);
        try {
            const res: IngestResponse = await ingestDocuments();
            setStatusMsg({
                text: `Ingested ${res.documents_processed} doc(s) → ${res.chunks_created} chunks`,
                ok: true,
            });
            onEngineReady();
        } catch (err: any) {
            setStatusMsg({ text: err.message ?? "Ingestion failed", ok: false });
        } finally {
            setIngesting(false);
        }
    };

    /* ---- drag & drop ------------------------------------------------------ */
    const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); };
    const onDragLeave = () => setDragOver(false);
    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
    };

    return (
        <>
            {open && (
                <div
                    className="fixed inset-0 bg-black/50 z-30 lg:hidden"
                    onClick={onClose}
                />
            )}

            <aside
                className={`fixed z-40 top-0 left-0 h-full w-72 bg-[var(--card)] border-r border-[var(--border)] flex flex-col transition-transform duration-200 lg:relative lg:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"
                    }`}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold text-sm">
                            NV
                        </div>
                        <span className="font-semibold text-sm">NeuraVault</span>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 rounded hover:bg-[var(--card-hover)] lg:hidden"
                    >
                        <X size={18} />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-3 py-4 space-y-4">
                    {/* Status cards */}
                    <StatusItem
                        icon={<Database size={16} />}
                        label="Vector DB"
                        value={engineReady ? "Loaded" : "Not loaded"}
                        ok={!!engineReady}
                    />
                    <StatusItem
                        icon={<FileText size={16} />}
                        label="LLM Model"
                        value="llama3.2 (Ollama)"
                        ok={!!engineReady}
                    />

                    {/* Separator */}
                    <div className="border-t border-[var(--border)]" />

                    {/* Upload zone */}
                    <div className="space-y-2">
                        <p className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                            Document Manager
                        </p>

                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf"
                            multiple
                            className="hidden"
                            onChange={(e) => e.target.files && handleFiles(e.target.files)}
                        />

                        <div
                            onDragOver={onDragOver}
                            onDragLeave={onDragLeave}
                            onDrop={onDrop}
                            onClick={() => fileInputRef.current?.click()}
                            className={`cursor-pointer rounded-xl border-2 border-dashed p-4 text-center transition-colors ${dragOver
                                    ? "border-brand-500 bg-brand-600/10"
                                    : "border-[var(--border)] hover:border-brand-500/50 hover:bg-[var(--card-hover)]"
                                }`}
                        >
                            {uploading ? (
                                <Loader2 size={24} className="mx-auto animate-spin text-brand-400" />
                            ) : (
                                <FolderUp size={24} className="mx-auto text-[var(--muted)]" />
                            )}
                            <p className="mt-2 text-xs text-[var(--muted)]">
                                {uploading ? "Uploading…" : "Drop PDFs here or click to browse"}
                            </p>
                        </div>
                    </div>

                    {/* Document list */}
                    {docs.length > 0 && (
                        <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                                <p className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">
                                    Documents ({docs.length})
                                </p>
                                <button onClick={refreshDocs} className="p-1 rounded hover:bg-[var(--card-hover)]" title="Refresh">
                                    <RefreshCw size={12} className="text-[var(--muted)]" />
                                </button>
                            </div>
                            <ul className="space-y-1 max-h-40 overflow-y-auto">
                                {docs.map((d) => (
                                    <li
                                        key={d.name}
                                        className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-[var(--background)] text-xs"
                                    >
                                        <File size={14} className="shrink-0 text-brand-400" />
                                        <span className="truncate flex-1" title={d.name}>{d.name}</span>
                                        <span className="text-[var(--muted)] whitespace-nowrap">{d.size_kb} KB</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Ingest button */}
                    <button
                        onClick={handleIngest}
                        disabled={ingesting || docs.length === 0}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-white text-sm font-medium"
                    >
                        {ingesting ? (
                            <Loader2 size={16} className="animate-spin" />
                        ) : (
                            <Database size={16} />
                        )}
                        {ingesting ? "Ingesting…" : "Ingest Documents"}
                    </button>

                    {/* Status message */}
                    {statusMsg && (
                        <div
                            className={`flex items-start gap-2 px-3 py-2 rounded-lg text-xs ${statusMsg.ok
                                    ? "bg-green-900/30 text-green-400"
                                    : "bg-red-900/30 text-red-400"
                                }`}
                        >
                            {statusMsg.ok ? <CheckCircle2 size={14} className="shrink-0 mt-0.5" /> : <AlertCircle size={14} className="shrink-0 mt-0.5" />}
                            <span>{statusMsg.text}</span>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-4 py-3 border-t border-[var(--border)] text-[10px] text-[var(--muted)] text-center">
                    NeuraVault v2.0 — 100 % offline
                </div>
            </aside>
        </>
    );
}

function StatusItem({
    icon,
    label,
    value,
    ok,
}: {
    icon: React.ReactNode;
    label: string;
    value: string;
    ok: boolean;
}) {
    return (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--background)] text-xs">
            <span className={ok ? "text-green-400" : "text-yellow-400"}>{icon}</span>
            <div className="min-w-0 flex-1">
                <p className="font-medium truncate">{label}</p>
                <p className="text-[var(--muted)] truncate">{value}</p>
            </div>
        </div>
    );
}
