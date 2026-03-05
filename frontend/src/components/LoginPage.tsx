"use client";

import { useState } from "react";
import { useAuth } from "@/components/AuthContext";
import { Lock, User, Mail, Eye, EyeOff } from "lucide-react";

export default function LoginPage() {
    const { login, register } = useAuth();
    const [mode, setMode] = useState<"login" | "register">("login");

    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [showPw, setShowPw] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setLoading(true);
        try {
            if (mode === "login") {
                await login(username, password);
            } else {
                await register(username, email, password);
            }
        } catch (err: any) {
            let msg = err.message ?? "Something went wrong.";
            // Try to parse JSON detail from FastAPI
            try {
                const parsed = JSON.parse(msg);
                msg = parsed.detail ?? msg;
            } catch { }
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-[var(--background)]">
            <div className="w-full max-w-md p-8 space-y-6 bg-[var(--card)] rounded-2xl border border-[var(--border)] shadow-xl">
                {/* Logo */}
                <div className="text-center space-y-2">
                    <div className="w-14 h-14 mx-auto rounded-xl bg-brand-600/20 flex items-center justify-center">
                        <span className="text-2xl font-bold text-brand-400">NV</span>
                    </div>
                    <h1 className="text-xl font-semibold">NeuraVault</h1>
                    <p className="text-sm text-[var(--muted)]">
                        {mode === "login" ? "Sign in to continue" : "Create your account"}
                    </p>
                </div>

                {/* Error */}
                {error && (
                    <div className="p-3 text-sm rounded-lg bg-red-900/30 text-red-400 border border-red-800">
                        {error}
                    </div>
                )}

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* Username */}
                    <div className="relative">
                        <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                        <input
                            type="text"
                            placeholder="Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            minLength={3}
                            maxLength={30}
                            className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)] text-sm focus:outline-none focus:ring-2 focus:ring-brand-600"
                        />
                    </div>

                    {/* Email (register only) */}
                    {mode === "register" && (
                        <div className="relative">
                            <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                            <input
                                type="email"
                                placeholder="Email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)] text-sm focus:outline-none focus:ring-2 focus:ring-brand-600"
                            />
                        </div>
                    )}

                    {/* Password */}
                    <div className="relative">
                        <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" />
                        <input
                            type={showPw ? "text" : "password"}
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            minLength={mode === "register" ? 8 : 1}
                            maxLength={128}
                            className="w-full pl-10 pr-10 py-2.5 rounded-xl bg-[var(--background)] border border-[var(--border)] text-sm focus:outline-none focus:ring-2 focus:ring-brand-600"
                        />
                        <button
                            type="button"
                            onClick={() => setShowPw((v) => !v)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--muted)] hover:text-white"
                            tabIndex={-1}
                        >
                            {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                    </div>

                    {mode === "register" && (
                        <p className="text-xs text-[var(--muted)]">
                            Password must be 8+ chars with uppercase, lowercase, digit & special character.
                        </p>
                    )}

                    {/* Submit */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-sm font-medium transition-colors disabled:opacity-50"
                    >
                        {loading
                            ? "Please wait…"
                            : mode === "login"
                                ? "Sign In"
                                : "Create Account"}
                    </button>
                </form>

                {/* Toggle mode */}
                <p className="text-center text-sm text-[var(--muted)]">
                    {mode === "login" ? (
                        <>
                            Don&apos;t have an account?{" "}
                            <button
                                onClick={() => { setMode("register"); setError(null); }}
                                className="text-brand-400 hover:underline"
                            >
                                Sign Up
                            </button>
                        </>
                    ) : (
                        <>
                            Already have an account?{" "}
                            <button
                                onClick={() => { setMode("login"); setError(null); }}
                                className="text-brand-400 hover:underline"
                            >
                                Sign In
                            </button>
                        </>
                    )}
                </p>
            </div>
        </div>
    );
}
