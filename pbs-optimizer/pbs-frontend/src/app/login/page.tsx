"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plane, Loader2, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const supabase = createClient();
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      setError(error.message);
      setLoading(false);
    } else {
      router.push("/dashboard");
      router.refresh();
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);

    const supabase = createClient();
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) {
      setError(error.message);
    } else {
      setMessage("Check your email for a confirmation link.");
    }
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-navy px-6">
      {/* Background effects */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute top-1/3 left-1/2 h-[500px] w-[500px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber/5 blur-[120px]" />
      </div>

      <div className="relative w-full max-w-sm">
        {/* Back to home */}
        <Link
          href="/"
          className="mb-8 inline-flex items-center gap-2 text-sm text-slate-warm/60 transition-colors hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>

        {/* Logo */}
        <div className="mb-8 flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber/10">
            <Plane className="h-5 w-5 text-amber" />
          </div>
          <span className="text-xl font-semibold tracking-tight text-white">
            Bid<span className="text-amber">Line</span>
          </span>
        </div>

        {/* Mode Toggle */}
        <div className="mb-6 flex rounded-lg border border-white/5 bg-navy-light p-1">
          <button
            onClick={() => {
              setMode("login");
              setError(null);
              setMessage(null);
            }}
            className={`flex-1 rounded-md py-2 text-sm font-medium transition-all ${
              mode === "login"
                ? "bg-navy-mid text-white"
                : "text-slate-warm/60 hover:text-white"
            }`}
          >
            Log In
          </button>
          <button
            onClick={() => {
              setMode("signup");
              setError(null);
              setMessage(null);
            }}
            className={`flex-1 rounded-md py-2 text-sm font-medium transition-all ${
              mode === "signup"
                ? "bg-navy-mid text-white"
                : "text-slate-warm/60 hover:text-white"
            }`}
          >
            Sign Up
          </button>
        </div>

        {/* Heading */}
        <h1 className="mb-2 text-2xl font-bold text-white">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h1>
        <p className="mb-6 text-sm text-slate-warm/60">
          {mode === "login"
            ? "Enter your credentials to access your dashboard."
            : "Start optimizing your PBS bids today."}
        </p>

        {/* Form */}
        <form onSubmit={mode === "login" ? handleLogin : handleSignup}>
          <div className="space-y-4">
            <div>
              <label
                htmlFor="email"
                className="mb-1.5 block text-sm font-medium text-slate-warm"
              >
                Email
              </label>
              <Input
                id="email"
                type="email"
                placeholder="you@airline.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="border-white/10 bg-navy-light text-white placeholder:text-slate-warm/40 focus:border-amber/50 focus:ring-amber/20"
              />
            </div>
            <div>
              <label
                htmlFor="password"
                className="mb-1.5 block text-sm font-medium text-slate-warm"
              >
                Password
              </label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className="border-white/10 bg-navy-light text-white placeholder:text-slate-warm/40 focus:border-amber/50 focus:ring-amber/20"
              />
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}

          {/* Success Message */}
          {message && (
            <div className="mt-4 rounded-lg border border-green-500/20 bg-green-500/10 px-4 py-3 text-sm text-green-400">
              {message}
            </div>
          )}

          {/* Submit */}
          <Button
            type="submit"
            disabled={loading}
            className="mt-6 w-full bg-amber font-medium text-navy hover:bg-amber-light disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : mode === "login" ? (
              "Log In"
            ) : (
              "Create Account"
            )}
          </Button>
        </form>

        {/* Footer */}
        <p className="mt-8 text-center text-xs text-slate-warm/40">
          By continuing, you agree to BidLine&apos;s Terms of Service and
          Privacy Policy.
        </p>
      </div>
    </div>
  );
}
