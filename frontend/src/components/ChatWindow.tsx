"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
    queryRAG,
    fetchStarters,
    fetchHealth,
    uploadFiles,
    ingestDocuments,
    type QueryResponse,
    type StarterQuestion,
} from "@/lib/api";
import MessageBubble, { type ChatMessage } from "@/components/MessageBubble";
import StarterQuestions from "@/components/StarterQuestions";
import Sidebar from "@/components/Sidebar";
import { useAuth } from "@/components/AuthContext";
import { Send, Upload, Menu, LogOut } from "lucide-react";

export default function ChatWindow() {
    /* ---- state ------------------------------------------------------------ */
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [starters, setStarters] = useState<StarterQuestion[]>([]);
    const [engineReady, setEngineReady] = useState<boolean | null>(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const { username, role, logout } = useAuth();
    const bottomRef = useRef<HTMLDivElement>(null);
    const fileRef = useRef<HTMLInputElement>(null);

    /* ---- effects ---------------------------------------------------------- */
    useEffect(() => {
        fetchStarters()
            .then(setStarters)
            .catch(() => { });
        fetchHealth()
            .then((h) => setEngineReady(h.engine_ready))
            .catch(() => setEngineReady(false));
    }, []);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    /* ---- handlers --------------------------------------------------------- */
    const sendMessage = useCallback(
        async (text?: string) => {
            const question = (text ?? input).trim();
            if (!question || loading) return;

            setInput("");
            const userMsg: ChatMessage = { role: "user", content: question };
            setMessages((prev) => [...prev, userMsg]);
            setLoading(true);

            try {
                const res: QueryResponse = await queryRAG(question);
                const assistantMsg: ChatMessage = {
                    role: "assistant",
                    content: res.answer,
                    sources: res.sources,
                };
                setMessages((prev) => [...prev, assistantMsg]);
            } catch (err: any) {
                const errMsg: ChatMessage = {
                    role: "assistant",
                    content: `**Error:** ${err.message ?? "Something went wrong."}`,
                };
                setMessages((prev) => [...prev, errMsg]);
            } finally {
                setLoading(false);
            }
        },
        [input, loading]
    );

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const sysMsg: ChatMessage = {
            role: "assistant",
            content: `Uploading **${files.length}** file(s)…`,
        };
        setMessages((prev) => [...prev, sysMsg]);
        setLoading(true);

        try {
            const uploadRes = await uploadFiles(files);
            const ingestRes = await ingestDocuments();
            const doneMsg: ChatMessage = {
                role: "assistant",
                content: `✅ Upload complete! Processed **${ingestRes.documents_processed}** document(s) into **${ingestRes.chunks_created}** chunks.\n\nYou can now ask questions about the new documents.`,
            };
            setMessages((prev) => [...prev, doneMsg]);
            setEngineReady(true);
        } catch (err: any) {
            const errMsg: ChatMessage = {
                role: "assistant",
                content: `**Upload failed:** ${err.message}`,
            };
            setMessages((prev) => [...prev, errMsg]);
        } finally {
            setLoading(false);
            if (fileRef.current) fileRef.current.value = "";
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    /* ---- render ----------------------------------------------------------- */
    return (
        <div className="flex w-full h-full">
            {/* Sidebar */}
            <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} engineReady={engineReady} onEngineReady={() => setEngineReady(true)} />

            <div className="flex flex-col flex-1 h-full">
                {/* Header */}
                <header className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)] bg-[var(--card)]">
                    <button
                        onClick={() => setSidebarOpen((v) => !v)}
                        className="p-1.5 rounded-lg hover:bg-[var(--card-hover)] transition-colors lg:hidden"
                    >
                        <Menu size={20} />
                    </button>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center text-white font-bold text-sm">
                            NV
                        </div>
                        <div>
                            <h1 className="text-sm font-semibold leading-tight">NeuraVault</h1>
                            <p className="text-xs text-[var(--muted)]">
                                Privacy-first forensic RAG
                            </p>
                        </div>
                    </div>
                    <div className="ml-auto flex items-center gap-2">
                        {engineReady !== null && (
                            <span
                                className={`text-xs px-2 py-0.5 rounded-full ${engineReady
                                    ? "bg-green-900/40 text-green-400"
                                    : "bg-yellow-900/40 text-yellow-400"
                                    }`}
                            >
                                {engineReady ? "Engine Ready" : "Engine Not Ready"}
                            </span>
                        )}
                        {username && (
                            <span className="text-xs text-[var(--muted)] hidden sm:inline">
                                {username}{role === "admin" ? " (admin)" : ""}
                            </span>
                        )}
                        <button
                            onClick={logout}
                            title="Sign out"
                            className="p-1.5 rounded-lg hover:bg-[var(--card-hover)] transition-colors text-[var(--muted)] hover:text-red-400"
                        >
                            <LogOut size={16} />
                        </button>
                    </div>
                </header>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
                    {messages.length === 0 && !loading && (
                        <div className="flex flex-col items-center justify-center h-full gap-6">
                            <div className="text-center space-y-2">
                                <div className="w-16 h-16 mx-auto rounded-2xl bg-brand-600/20 flex items-center justify-center">
                                    <span className="text-3xl font-bold text-brand-400">NV</span>
                                </div>
                                <h2 className="text-xl font-semibold">
                                    Welcome to NeuraVault
                                </h2>
                                <p className="text-sm text-[var(--muted)] max-w-md">
                                    Ask any question about the documents in your vault. All
                                    processing happens locally — your data never leaves this
                                    machine.
                                </p>
                            </div>
                            {starters.length > 0 && (
                                <StarterQuestions
                                    starters={starters}
                                    onSelect={(msg) => sendMessage(msg)}
                                />
                            )}
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <MessageBubble key={i} message={msg} />
                    ))}

                    {loading && (
                        <div className="flex items-start gap-3 max-w-3xl mx-auto">
                            <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center text-white text-xs font-bold shrink-0">
                                NV
                            </div>
                            <div className="px-4 py-3 rounded-2xl bg-[var(--card)] text-sm dot-animation">
                                <span>●</span> <span>●</span> <span>●</span>
                            </div>
                        </div>
                    )}

                    <div ref={bottomRef} />
                </div>

                {/* Input bar */}
                <div className="border-t border-[var(--border)] bg-[var(--card)] px-4 py-3">
                    <form
                        onSubmit={(e) => {
                            e.preventDefault();
                            sendMessage();
                        }}
                        className="flex items-end gap-2 max-w-3xl mx-auto"
                    >
                        {/* hidden file input */}
                        <input
                            ref={fileRef}
                            type="file"
                            accept=".pdf"
                            multiple
                            className="hidden"
                            onChange={handleUpload}
                        />
                        <button
                            type="button"
                            onClick={() => fileRef.current?.click()}
                            title="Upload PDF"
                            className="p-2.5 rounded-xl hover:bg-[var(--card-hover)] transition-colors text-[var(--muted)] hover:text-white"
                        >
                            <Upload size={18} />
                        </button>

                        <textarea
                            rows={1}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask a question about your documents…"
                            className="flex-1 resize-none rounded-xl bg-[var(--background)] border border-[var(--border)] px-4 py-2.5 text-sm placeholder:text-[var(--muted)] focus:outline-none focus:ring-2 focus:ring-brand-500/50"
                            style={{ maxHeight: 160 }}
                        />

                        <button
                            type="submit"
                            disabled={loading || !input.trim()}
                            className="p-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-white"
                        >
                            <Send size={18} />
                        </button>
                    </form>
                    <p className="text-center text-[10px] text-[var(--muted)] mt-2">
                        100 % offline · All data stays on your machine
                    </p>
                </div>
            </div>
        </div>
    );
}
