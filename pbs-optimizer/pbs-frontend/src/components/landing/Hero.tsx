"use client";

import { Button } from "@/components/ui/button";
import { ArrowRight, Shield, Zap, BarChart3 } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section className="relative min-h-screen overflow-hidden bg-navy pt-16">
      {/* Background Effects */}
      <div className="pointer-events-none absolute inset-0">
        {/* Radial glow */}
        <div className="absolute top-1/4 left-1/2 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber/5 blur-[120px]" />
        {/* Grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
            backgroundSize: "60px 60px",
          }}
        />
      </div>

      <div className="relative mx-auto flex max-w-7xl flex-col items-center px-6 pt-24 text-center md:pt-36">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 inline-flex items-center gap-2 rounded-full border border-amber/20 bg-amber/5 px-4 py-1.5"
        >
          <Zap className="h-3.5 w-3.5 text-amber" />
          <span className="text-xs font-medium tracking-wide text-amber">
            AI-POWERED PBS OPTIMIZATION
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="max-w-4xl text-4xl leading-[1.1] font-bold tracking-tight text-white md:text-6xl lg:text-7xl"
        >
          Bid smarter.{" "}
          <span className="bg-gradient-to-r from-amber to-amber-light bg-clip-text text-transparent">
            Every month.
          </span>
        </motion.h1>

        {/* Subheadline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-warm md:text-xl"
        >
          Tell BidLine what matters to you — days off, layovers, pay — in plain
          English. Our AI analyzes every sequence and ranks them so you can bid
          with confidence.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-10 flex flex-col gap-4 sm:flex-row"
        >
          <Link href="/login">
            <Button
              size="lg"
              className="bg-amber px-8 text-base font-semibold text-navy hover:bg-amber-light"
            >
              Start Optimizing
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
          <Link href="#how-it-works">
            <Button
              size="lg"
              variant="outline"
              className="border-white/10 px-8 text-base text-white hover:bg-white/5"
            >
              See How It Works
            </Button>
          </Link>
        </motion.div>

        {/* Trust Indicators */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mt-16 flex flex-wrap items-center justify-center gap-8 text-sm text-slate-warm/60"
        >
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            <span>Secure &amp; Private</span>
          </div>
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            <span>97% Parse Accuracy</span>
          </div>
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            <span>Results in Seconds</span>
          </div>
        </motion.div>

        {/* Preview Window */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.6 }}
          className="mt-20 w-full max-w-4xl"
        >
          <div className="rounded-xl border border-white/10 bg-navy-light p-1 shadow-2xl shadow-black/40">
            {/* Window Chrome */}
            <div className="flex items-center gap-2 border-b border-white/5 px-4 py-3">
              <div className="h-3 w-3 rounded-full bg-red-500/60" />
              <div className="h-3 w-3 rounded-full bg-yellow-500/60" />
              <div className="h-3 w-3 rounded-full bg-green-500/60" />
              <span className="ml-3 text-xs text-slate-warm/40">
                bidline.app/dashboard
              </span>
            </div>

            {/* Mock Dashboard Content */}
            <div className="p-6">
              {/* Input */}
              <div className="rounded-lg border border-white/5 bg-navy/60 p-4">
                <p className="mb-2 text-xs font-medium tracking-wide text-slate-warm/60">
                  YOUR PREFERENCES
                </p>
                <p className="text-sm text-white/80">
                  &quot;I want Fridays off, Hawaii layovers, no redeyes, and
                  maximize my pay&quot;
                </p>
              </div>

              {/* Results Preview */}
              <div className="mt-4 space-y-3">
                {/* Sequence 1 */}
                <div className="flex items-center gap-4 rounded-lg border border-white/5 bg-navy/40 p-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber/10 text-sm font-bold text-amber">
                    #1
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white">
                        SEQ 217
                      </span>
                      <span className="rounded bg-green-500/10 px-2 py-0.5 text-xs text-green-400">
                        73.6
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-warm/60">
                      No conflicts · Day trip · Tue/Sat
                    </p>
                  </div>
                  <div className="hidden h-2 w-32 overflow-hidden rounded-full bg-white/5 sm:block">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber to-amber-light"
                      style={{ width: "73.6%" }}
                    />
                  </div>
                </div>

                {/* Sequence 2 */}
                <div className="flex items-center gap-4 rounded-lg border border-white/5 bg-navy/40 p-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/5 text-sm font-bold text-slate-warm">
                    #2
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white">
                        SEQ 224
                      </span>
                      <span className="rounded bg-amber/10 px-2 py-0.5 text-xs text-amber">
                        68.2
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-warm/60">
                      OGG layover ✓ · Works Friday ✗ · Redeye ✗
                    </p>
                  </div>
                  <div className="hidden h-2 w-32 overflow-hidden rounded-full bg-white/5 sm:block">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber to-amber-light"
                      style={{ width: "68.2%" }}
                    />
                  </div>
                </div>

                {/* Sequence 3 */}
                <div className="flex items-center gap-4 rounded-lg border border-white/5 bg-navy/40 p-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/5 text-sm font-bold text-slate-warm">
                    #3
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-white">
                        SEQ 218
                      </span>
                      <span className="rounded bg-white/5 px-2 py-0.5 text-xs text-slate-warm">
                        66.1
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-warm/60">
                      Works Friday ✗ · No layover
                    </p>
                  </div>
                  <div className="hidden h-2 w-32 overflow-hidden rounded-full bg-white/5 sm:block">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-amber to-amber-light"
                      style={{ width: "66.1%" }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Fade out at bottom */}
          <div className="relative -mt-20 h-20 bg-gradient-to-t from-navy to-transparent" />
        </motion.div>
      </div>
    </section>
  );
}
