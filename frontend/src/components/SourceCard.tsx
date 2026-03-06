"use client";

import { useState } from "react";
import type { SourceInfo } from "@/lib/api";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";

interface Props {
    index: number;
    source: SourceInfo;
}

export default function SourceCard({ index, source }: Props) {
    const [open, setOpen] = useState(false);

    return (
        <div className="border border-[var(--border)] rounded-xl bg-[var(--card)] overflow-hidden">
            <button
                onClick={() => setOpen((v) => !v)}
                className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-[var(--card-hover)] transition-colors"
            >
                <FileText size={14} className="text-brand-400 shrink-0" />
                <span className="text-xs font-medium truncate flex-1">
                    Source {index}: {source.filename}
                </span>
                <span className="text-[10px] text-[var(--muted)] uppercase">
                    {source.document_type}
                </span>
                {open ? (
                    <ChevronUp size={14} className="text-[var(--muted)]" />
                ) : (
                    <ChevronDown size={14} className="text-[var(--muted)]" />
                )}
            </button>

            {open && (
                <div className="px-3 pb-3 pt-1 text-xs text-[var(--muted)] leading-relaxed border-t border-[var(--border)]">
                    <p className="italic">&ldquo;{source.excerpt}&rdquo;</p>
                </div>
            )}
        </div>
    );
}
