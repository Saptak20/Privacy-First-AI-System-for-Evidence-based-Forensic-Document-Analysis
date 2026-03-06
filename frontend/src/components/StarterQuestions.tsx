"use client";

import type { StarterQuestion } from "@/lib/api";
import { MessageCircleQuestion } from "lucide-react";

interface Props {
    starters: StarterQuestion[];
    onSelect: (message: string) => void;
}

export default function StarterQuestions({ starters, onSelect }: Props) {
    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
            {starters.map((s) => (
                <button
                    key={s.label}
                    onClick={() => onSelect(s.message)}
                    className="flex items-start gap-2 text-left px-4 py-3 rounded-xl border border-[var(--border)] bg-[var(--card)] hover:bg-[var(--card-hover)] transition-colors"
                >
                    <MessageCircleQuestion
                        size={16}
                        className="text-brand-400 mt-0.5 shrink-0"
                    />
                    <div>
                        <p className="text-sm font-medium">{s.label}</p>
                        <p className="text-xs text-[var(--muted)] line-clamp-2">
                            {s.message}
                        </p>
                    </div>
                </button>
            ))}
        </div>
    );
}
