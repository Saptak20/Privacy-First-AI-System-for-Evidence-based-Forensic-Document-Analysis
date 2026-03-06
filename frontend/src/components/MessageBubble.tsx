"use client";

import ReactMarkdown from "react-markdown";
import SourceCard from "@/components/SourceCard";
import type { SourceInfo } from "@/lib/api";
import { User, Bot } from "lucide-react";

export interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    sources?: SourceInfo[];
}

interface Props {
    message: ChatMessage;
}

export default function MessageBubble({ message }: Props) {
    const isUser = message.role === "user";

    return (
        <div className={`flex items-start gap-3 max-w-3xl mx-auto ${isUser ? "flex-row-reverse" : ""}`}>
            {/* Avatar */}
            <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center text-white text-xs font-bold shrink-0 ${isUser ? "bg-gray-600" : "bg-brand-600"
                    }`}
            >
                {isUser ? <User size={14} /> : "NV"}
            </div>

            {/* Content */}
            <div className={`space-y-2 max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
                <div
                    className={`px-4 py-3 rounded-2xl text-sm prose ${isUser
                            ? "bg-brand-600 text-white rounded-tr-md"
                            : "bg-[var(--card)] rounded-tl-md"
                        }`}
                >
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>

                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                    <div className="space-y-2 w-full">
                        <p className="text-xs font-medium text-[var(--muted)] pl-1">
                            Sources ({message.sources.length})
                        </p>
                        {message.sources.map((src, i) => (
                            <SourceCard key={i} index={i + 1} source={src} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
