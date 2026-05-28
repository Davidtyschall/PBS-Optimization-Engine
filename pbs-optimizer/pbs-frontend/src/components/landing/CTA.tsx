"use client";

import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";

export default function CTA() {
  return (
    <section className="bg-navy-light py-24">
      <div className="mx-auto max-w-7xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="relative overflow-hidden rounded-2xl border border-amber/10 bg-gradient-to-br from-navy-mid to-navy p-12 text-center md:p-20"
        >
          {/* Glow effect */}
          <div className="pointer-events-none absolute top-0 left-1/2 h-40 w-80 -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber/10 blur-[80px]" />

          <h2 className="relative text-3xl font-bold text-white md:text-5xl">
            Stop guessing your bid.
          </h2>
          <p className="relative mt-4 text-lg text-slate-warm">
            Join pilots who bid smarter with AI-powered optimization.
          </p>
          <div className="relative mt-8">
            <Link href="/login">
              <Button
                size="lg"
                className="bg-amber px-10 text-base font-semibold text-navy hover:bg-amber-light"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
          <p className="relative mt-4 text-sm text-slate-warm/50">
            No credit card required
          </p>
        </motion.div>
      </div>
    </section>
  );
}
