"use client";

import { motion } from "framer-motion";

const steps = [
  {
    number: "01",
    title: "Upload Your Bid Packet",
    description:
      "Drop in your monthly PBS bid packet PDF. Our parser extracts every sequence with 97% accuracy.",
  },
  {
    number: "02",
    title: "Tell Us Your Preferences",
    description:
      "Type what matters to you in plain English. Days off, layover cities, avoid redeyes, maximize pay — whatever you need.",
  },
  {
    number: "03",
    title: "Get Ranked Results",
    description:
      "Every sequence is scored and ranked against your preferences. See exactly why each one placed where it did.",
  },
  {
    number: "04",
    title: "Bid With Confidence",
    description:
      "Use the ranked list to guide your PBS bid. No more guessing, no more spreadsheets, no more missed opportunities.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="bg-navy py-24">
      <div className="mx-auto max-w-7xl px-6">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <p className="text-sm font-medium tracking-wide text-amber">
            HOW IT WORKS
          </p>
          <h2 className="mt-3 text-3xl font-bold text-white md:text-4xl">
            From preferences to ranked bids in minutes
          </h2>
        </motion.div>

        {/* Steps */}
        <div className="mt-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15 }}
              className="relative"
            >
              {/* Connector line */}
              {i < steps.length - 1 && (
                <div className="absolute top-8 right-0 hidden h-px w-full translate-x-1/2 bg-gradient-to-r from-amber/30 to-transparent lg:block" />
              )}

              <div className="relative">
                <span className="text-4xl font-bold text-amber/20">
                  {step.number}
                </span>
                <h3 className="mt-3 text-lg font-semibold text-white">
                  {step.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-warm">
                  {step.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
