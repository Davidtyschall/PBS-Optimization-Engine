import { Plane } from "lucide-react";
import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-white/5 bg-navy py-12">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-center justify-between gap-6 md:flex-row">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber/10">
              <Plane className="h-4 w-4 text-amber" />
            </div>
            <span className="text-lg font-semibold tracking-tight text-white">
              Bid<span className="text-amber">Line</span>
            </span>
          </div>

          {/* Links */}
          <div className="flex gap-8 text-sm text-slate-warm/60">
            <Link
              href="/pricing"
              className="transition-colors hover:text-white"
            >
              Pricing
            </Link>
            <Link href="#" className="transition-colors hover:text-white">
              Privacy
            </Link>
            <Link href="#" className="transition-colors hover:text-white">
              Terms
            </Link>
            <Link href="#" className="transition-colors hover:text-white">
              Contact
            </Link>
          </div>

          {/* Copyright */}
          <p className="text-sm text-slate-warm/40">
            &copy; {new Date().getFullYear()} BidLine. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
