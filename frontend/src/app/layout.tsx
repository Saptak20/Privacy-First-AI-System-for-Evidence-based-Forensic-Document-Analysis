import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "NeuraVault — Privacy-First RAG",
    description:
        "Ask questions about your PDF documents with full source citations, 100 % offline.",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
                {children}
            </body>
        </html>
    );
}
