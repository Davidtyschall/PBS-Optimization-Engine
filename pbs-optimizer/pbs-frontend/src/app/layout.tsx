import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  title: "BidLine | Next-generation PBS Optimization",
  description:
    "Stop guessing your PBS bids. BidLine uses AI to analyze your preferences and rank every sequence — so you bid smarter, every month.",
  keywords: [
    "PBS",
    "pilot bidding",
    "airline scheduling",
    "crew scheduling",
    "preferential bidding",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geist.variable} ${geistMono.variable} font-[family-name:var(--font-geist)] antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
