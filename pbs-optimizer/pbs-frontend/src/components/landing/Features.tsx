"use client";

import {
  MessageSquare,
  BarChart3,
  Eye,
  Clock,
  Shield,
  Smartphone,
} from "lucide-react";
import { motion } from "framer-motion";

const features = [
  {
    icon: MessageSquare,
    title: "Natural Language Input",
    description:
      "Just type what you want: \"Fridays off, Hawaii trips, no redeyes.\" Our AI understands pilot preferences in plain English.",
  },
  {
    icon: BarChart3,
    title: "Intelligent Scoring",
    description:
      "Every sequence is scored across days off, layovers, times, trip length, and credit — weighted to your priorities.",
  },
  {
    icon: Eye,
    title: "Full Explainability",
    description:
      "See exactly why each sequence ranked where it did. No black boxes. Every point gain and penalty is transparent.",
  },
  {
    icon: Clock,
    title: "Results in Seconds",
    description:
      "Parse your bid packet, enter your preferences, and get ranked results instantly. No more spreadsheet guesswork.",
  },
  {
    icon: Shield,
    title: "Secure & Private",
    description:
      "Your preferences and bid data are encrypted and never shared. Your scheduling strategy stays yours.",
  },
  {
    icon: Smartphone,
    title: "Works Anywhere",
    description:
      "Optimize your bid from your phone on a layover or your laptop at home. Fully responsive, no app download needed.",
  },
];

export default function Features() {
  return (
    <section id="features" className="bg-navy-light py-24">
      <div className="mx-auto max-w-7xl px-6">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <p className="text-sm font-medium tracking-wide text-amber">
            FEATURES
          </p>
          <h2 className="mt-3 text-3xl font-bold text-white md:text-4xl">
            Everything you need to bid with confidence
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-slate-warm">
            Built by aviation professionals who understand PBS inside and out.
          </p>
        </motion.div>

        {/* Feature Grid */}
        <div className="mt-16 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="group rounded-xl border border-white/5 bg-navy/60 p-6 transition-all duration-300 hover:border-amber/20 hover:bg-navy/80"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-amber/10 transition-colors group-hover:bg-amber/20">
                <feature.icon className="h-5 w-5 text-amber" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-white">
                {feature.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-warm">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
