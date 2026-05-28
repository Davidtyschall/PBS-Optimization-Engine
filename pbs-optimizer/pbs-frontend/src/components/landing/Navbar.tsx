"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu, Plane } from "lucide-react";
import Link from "next/link";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-navy/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber/10">
            <Plane className="h-4 w-4 text-amber" />
          </div>
          <span className="text-lg font-semibold tracking-tight text-white">
            Bid<span className="text-amber">Line</span>
          </span>
        </Link>

        {/* Desktop Nav */}
        <div className="hidden items-center gap-8 md:flex">
          <Link
            href="#features"
            className="text-sm text-slate-warm transition-colors hover:text-white"
          >
            Features
          </Link>
          <Link
            href="#how-it-works"
            className="text-sm text-slate-warm transition-colors hover:text-white"
          >
            How It Works
          </Link>
          <Link
            href="/pricing"
            className="text-sm text-slate-warm transition-colors hover:text-white"
          >
            Pricing
          </Link>
        </div>

        {/* Desktop CTA */}
        <div className="hidden items-center gap-3 md:flex">
          <Link href="/login">
            <Button
              variant="ghost"
              className="text-slate-warm hover:bg-white/5 hover:text-white"
            >
              Log In
            </Button>
          </Link>
          <Link href="/login">
            <Button className="bg-amber font-medium text-navy hover:bg-amber-light">
              Get Started
            </Button>
          </Link>
        </div>

        {/* Mobile Menu */}
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild className="md:hidden">
            <Button variant="ghost" size="icon" className="text-white">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="border-navy-mid bg-navy">
            <div className="mt-8 flex flex-col gap-6">
              <Link
                href="#features"
                onClick={() => setOpen(false)}
                className="text-lg text-slate-warm transition-colors hover:text-white"
              >
                Features
              </Link>
              <Link
                href="#how-it-works"
                onClick={() => setOpen(false)}
                className="text-lg text-slate-warm transition-colors hover:text-white"
              >
                How It Works
              </Link>
              <Link
                href="/pricing"
                onClick={() => setOpen(false)}
                className="text-lg text-slate-warm transition-colors hover:text-white"
              >
                Pricing
              </Link>
              <div className="mt-4 flex flex-col gap-3">
                <Link href="/login">
                  <Button
                    variant="outline"
                    className="w-full border-white/10 text-white hover:bg-white/5"
                  >
                    Log In
                  </Button>
                </Link>
                <Link href="/login">
                  <Button className="w-full bg-amber font-medium text-navy hover:bg-amber-light">
                    Get Started
                  </Button>
                </Link>
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </nav>
  );
}
