"use client";

import { useState } from "react";
import { fetchTimeline, type TimelineEvent } from "@/lib/api";
import { Clock, Loader2, X, AlertTriangle, CheckCircle } from "lucide-react";

interface Props {
    open: boolean;
    onClose: () => void;
}

export default function TimelinePanel({ open, onClose }: Props) {
    const [events, setEvents] = useState<TimelineEvent[]>([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [hasLoaded, setHasLoaded] = useState(false);

    const handleExtract = async () => {
        setLoading(true);
        setMessage(null);
        try {
            const res = await fetchTimeline();
            setEvents(res.events);
            setMessage(res.message);
            setHasLoaded(true);
        } catch (err: any) {
            setMessage(err.message ?? "Timeline extraction failed");
        } finally {
            setLoading(false);
        }
    };

    if (!open) return null;

    const confidenceColor = (c: string) => {
        switch (c) {
            case "high":
                return "text-green-400 bg-green-900/30";
            case "medium":
                return "text-yellow-400 bg-yellow-900/30";
            case "low":
                return "text-red-400 bg-red-900/30";
            default:
                return "text-gray-400 bg-gray-900/30";
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="w-full max-w-2xl max-h-[80vh] bg-[var(--card)] border border-[var(--border)] rounded-2xl flex flex-col overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
                    <div className="flex items-center gap-2">
                        <Clock size={18} className="text-brand-400" />
                        <h2 className="text-sm font-semibold">Timeline Extraction</h2>
                    </div>
                    <button onClick={onClose} className="p-1 rounded-lg hover:bg-[var(--card-hover)]">
                        <X size={18} />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
                    {!hasLoaded && !loading && (
                        <div className="text-center space-y-3 py-8">
                            <Clock size={40} className="mx-auto text-brand-400/50" />
                            <p className="text-sm text-[var(--muted)]">
                                Extract chronological events and dates from your ingested documents.
                            </p>
                            <button
                                onClick={handleExtract}
                                className="px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-sm font-medium transition-colors"
                            >
                                Extract Timeline
                            </button>
                        </div>
                    )}

                    {loading && (
                        <div className="flex flex-col items-center gap-3 py-12">
                            <Loader2 size={32} className="animate-spin text-brand-400" />
                            <p className="text-sm text-[var(--muted)]">Analyzing documents for timeline events...</p>
                        </div>
                    )}

                    {hasLoaded && !loading && (
                        <>
                            {message && (
                                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-brand-600/10 text-brand-300 text-xs">
                                    <CheckCircle size={14} />
                                    <span>{message}</span>
                                </div>
                            )}

                            {events.length === 0 ? (
                                <div className="text-center py-8">
                                    <AlertTriangle size={32} className="mx-auto text-yellow-400/50 mb-2" />
                                    <p className="text-sm text-[var(--muted)]">No timeline events found in the documents.</p>
                                    <button
                                        onClick={handleExtract}
                                        className="mt-3 px-4 py-2 rounded-xl bg-[var(--card-hover)] text-sm hover:bg-brand-600/20 transition-colors"
                                    >
                                        Try Again
                                    </button>
                                </div>
                            ) : (
                                <div className="relative">
                                    {/* Timeline line */}
                                    <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-brand-600/30" />

                                    {events.map((ev, i) => (
                                        <div key={i} className="relative flex gap-4 pb-6 last:pb-0">
                                            {/* Dot */}
                                            <div className="relative z-10 w-8 h-8 rounded-full bg-brand-600/20 border-2 border-brand-500 flex items-center justify-center shrink-0 mt-0.5">
                                                <div className="w-2.5 h-2.5 rounded-full bg-brand-400" />
                                            </div>

                                            {/* Card */}
                                            <div className="flex-1 bg-[var(--background)] border border-[var(--border)] rounded-xl p-3 space-y-1.5">
                                                <div className="flex items-center gap-2 flex-wrap">
                                                    <span className="text-xs font-bold text-brand-300">{ev.date}</span>
                                                    <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${confidenceColor(ev.confidence)}`}>
                                                        {ev.confidence}
                                                    </span>
                                                </div>
                                                <p className="text-sm">{ev.event}</p>
                                                <p className="text-[11px] text-[var(--muted)]">Source: {ev.source}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {events.length > 0 && (
                                <button
                                    onClick={handleExtract}
                                    className="w-full mt-2 px-4 py-2 rounded-xl bg-[var(--card-hover)] text-sm hover:bg-brand-600/20 transition-colors text-[var(--muted)]"
                                >
                                    Re-analyze
                                </button>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
