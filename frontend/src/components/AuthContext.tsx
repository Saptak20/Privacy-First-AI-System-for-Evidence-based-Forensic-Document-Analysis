"use client";

import {
    createContext,
    useContext,
    useState,
    useEffect,
    useCallback,
    type ReactNode,
} from "react";
import {
    login as apiLogin,
    register as apiRegister,
    logout as apiLogout,
    isAuthenticated,
    getStoredUser,
    type TokenResponse,
} from "@/lib/api";

interface AuthState {
    authenticated: boolean;
    username: string | null;
    role: string | null;
}

interface AuthContextType extends AuthState {
    login: (username: string, password: string) => Promise<void>;
    register: (username: string, email: string, password: string) => Promise<void>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<AuthState>({
        authenticated: false,
        username: null,
        role: null,
    });

    // Hydrate from localStorage on mount
    useEffect(() => {
        if (isAuthenticated()) {
            const user = getStoredUser();
            setState({
                authenticated: true,
                username: user?.username ?? null,
                role: user?.role ?? null,
            });
        }
    }, []);

    const login = useCallback(async (username: string, password: string) => {
        const data: TokenResponse = await apiLogin(username, password);
        setState({ authenticated: true, username: data.username, role: data.role });
    }, []);

    const register = useCallback(
        async (username: string, email: string, password: string) => {
            const data: TokenResponse = await apiRegister(username, email, password);
            setState({ authenticated: true, username: data.username, role: data.role });
        },
        [],
    );

    const logout = useCallback(() => {
        apiLogout();
        setState({ authenticated: false, username: null, role: null });
    }, []);

    return (
        <AuthContext.Provider value={{ ...state, login, register, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth(): AuthContextType {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
    return ctx;
}
